# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import logging
import time
from typing import Any, Union, Literal, NoReturn
from uuid import uuid4
from pydantic import BaseModel

# Lungo_Improvement_Opt4: TTL cache for farm yield inventory responses.
# Farm yields change slowly (minutes to hours). Caching identical prompts for 60 seconds
# avoids a full A2A round-trip + LLM call for repeated identical queries.
# cachetools.TTLCache is O(1) get/set and auto-expires entries after `ttl` seconds.
from cachetools import TTLCache

_yield_cache: TTLCache = TTLCache(maxsize=256, ttl=60)
_cache_hits: int = 0
_cache_misses: int = 0
_cache_lock = asyncio.Lock()  # protects _cache_hits / _cache_misses counters

# Lungo_Improvement_Opt5: A2A client cache — reuse clients across calls.
# a2a_client_factory.create() is expensive: it negotiates a SLIM/NATS session.
# Keying by (farm_slug, transport) means the session is created once per farm
# and shared across all subsequent requests, removing per-request setup latency.
_a2a_client_cache: dict[tuple[str, str], Any] = {}

# Lungo_Improvement_Opt8: asyncio.Semaphore to cap concurrent A2A calls.
# Without a cap, a broadcast or burst of concurrent requests could overwhelm the
# SLIM transport layer with simultaneous sessions. The semaphore limits active
# A2A in-flight calls to 5, smoothing load without serialising requests.
_a2a_semaphore = asyncio.Semaphore(5)

def get_cache_stats() -> dict:
    """
    Lungo_Improvement_Opt4: Return live TTL cache statistics.
    Called by the /metrics/cache endpoint so the benchmark script and Grafana
    can observe cache behaviour without parsing logs.
    """
    total = _cache_hits + _cache_misses
    hit_rate = round((_cache_hits / total * 100), 1) if total > 0 else 0.0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate_pct": hit_rate,
        "cache_size": len(_yield_cache),
        "cache_maxsize": _yield_cache.maxsize,
        "cache_ttl_seconds": _yield_cache.ttl,
    }

from a2a.types import (
    AgentCard,
    SendMessageRequest,
    MessageSendParams,
    Message,
    Part,
    TextPart,
    Role,
)
from langchain_core.tools import tool, ToolException
from langchain_core.messages import AnyMessage, ToolMessage
from agntcy_app_sdk import InterfaceTransport, get_agent_identifier
from ioa_observe.sdk.decorators import tool as ioa_tool_decorator


from agents.supervisors.auction.graph.a2a_retry import (
    send_a2a_with_retry,
    TransportTimeoutError,
    RemoteAgentNoResponseError,
)
from agents.supervisors.auction.graph.models import (
    InventoryArgs,
    CreateOrderArgs,
)
from agents.supervisors.auction.graph.shared import a2a_client_factory, farm_registry
from config.config import (
    DEFAULT_MESSAGE_TRANSPORT,
    IDENTITY_API_KEY,
    IDENTITY_API_SERVER_URL,
)
from services.identity_service import IdentityService
from services.identity_service_impl import IdentityServiceImpl


logger = logging.getLogger("lungo.supervisor.tools")


class A2AAgentError(ToolException):
    """Custom exception for errors related to A2A agent communication or status."""
    pass


def _extract_text_from_events(events: list) -> str | None:
    """Extract the last text content from a list of A2A stream events.

    Handles both ``Message`` objects and ``ClientEvent`` tuples
    ``(Task, UpdateEvent)``.  Returns the last text found, or None.
    """
    result_text = None
    for response in events:
        if isinstance(response, Message):
            for part in response.parts:
                part_root = part.root
                if hasattr(part_root, "text"):
                    result_text = part_root.text.strip()
        elif isinstance(response, tuple):
            task_update, _event = response
            if hasattr(task_update, "status") and task_update.status and task_update.status.message:
                for part in task_update.status.message.parts:
                    part_root = part.root
                    if hasattr(part_root, "text"):
                        result_text = part_root.text.strip()
    return result_text


def tools_or_next(tools_node: str, end_node: str = "__end__"):
  """
  Returns a conditional function for LangGraph to determine the next node
  based on whether the last message contains tool calls.

  If the message includes tool calls, the workflow proceeds to the `tools_node`.
  If the message is a ToolMessage or has no tool calls, the workflow proceeds to `end_node`.

  Args:
    tools_node (str): The name of the node to route to if tool calls are detected.
    end_node (str, optional): The fallback node if no tool calls are found. Defaults to '__end__'.

  Returns:
    Callable: A function compatible with LangGraph conditional edge handling.
  """

  def custom_tools_condition_fn(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
  ) -> Literal[tools_node, end_node]: # type: ignore

    if isinstance(state, list):
      ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
      ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
      ai_message = messages[-1]
    else:
      raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if isinstance(ai_message, ToolMessage):
        logger.debug("Last message is a ToolMessage, returning end_node: %s", end_node)
        return end_node

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
      logger.debug("Last message has tool calls, returning tools_node: %s", tools_node)
      return tools_node

    logger.debug("Last message has no tool calls, returning end_node: %s", end_node)
    return end_node

  return custom_tools_condition_fn

def get_farm_card(farm: str) -> AgentCard | None:
    """
    Look up a farm's AgentCard by its canonical slug.

    Args:
        farm (str): The canonical slug of the farm (e.g., "brazil", "colombia", "vietnam").

    Returns:
        AgentCard | None: The matching AgentCard if found, otherwise None.
    """
    card = farm_registry.get(farm)
    if card is None:
        logger.error(
            "Unknown farm slug: '%s'. Registered farms: %s",
            farm, farm_registry.slugs(),
        )
    return card

def verify_farm_identity(identity_service: IdentityService, farm_name: str):
    """
    Verifies the identity of a farm by matching the farm name with the app name,
    retrieving the badge, and verifying it.

    Args:
        identity_service (IdentityServiceImpl): The identity service implementation.
        farm_name (str): The name of the farm to verify.

    Raises:
        A2AAgentError: If the app is not found or verification fails.
    """
    try:
        all_apps = identity_service.get_all_apps()
        matched_app = next((app for app in all_apps.apps if app.name.lower() == farm_name.lower()), None)

        if not matched_app:
            err_msg = f"No matching identity app service found, this farm does not have identity service enabled."
            logger.error(err_msg)
            raise A2AAgentError(err_msg)


        badge = identity_service.get_badge_for_app(matched_app.id)
        success = identity_service.verify_badges(badge)

        if success.get("status") is not True:
            raise A2AAgentError(f"Failed to verify badge.")

        logger.info(f"Verification successful for farm '{farm_name}'.")
    except Exception as e:
        raise A2AAgentError(e) # Re-raise as our custom exception

async def get_farm_yield_inventory(prompt: str, farm: str) -> str:
    """
    Fetch yield inventory from a specific farm.

    Args:
        prompt (str): The prompt to send to the farm to retrieve their yields
        farm (str): The farm to send the request to

    Returns:
        str: current yield amount

    Raises:
        A2AAgentError: If there's an issue with farm identification, communication, or the farm agent returns an error.
        ValueError: For invalid input arguments.
    """
    global _cache_hits, _cache_misses

    logger.info("entering get_farm_yield_inventory tool with prompt: %s, farm: %s", prompt, farm)
    if not farm:
        raise ValueError("No farm was provided. Please provide a farm to get the yield from.")

    card = get_farm_card(farm)
    if card is None:
        raise A2AAgentError(f"Farm '{farm}' not recognized. Available farms "
                             f"are: {', '.join(farm_registry.slugs())}.")

    # Lungo_Improvement_Opt4: check TTL cache before making an A2A call.
    # Cache key combines farm slug + normalised prompt so different questions don't collide.
    cache_key = (farm, prompt.strip().lower())
    cached = _yield_cache.get(cache_key)
    if cached is not None:
        _cache_hits += 1
        logger.info("[Lungo_Improvement_Opt4] Cache HIT for farm=%s  hits=%d misses=%d", farm, _cache_hits, _cache_misses)
        return cached
    _cache_misses += 1
    logger.info("[Lungo_Improvement_Opt4] Cache MISS for farm=%s  hits=%d misses=%d", farm, _cache_hits, _cache_misses)

    try:
        # Lungo_Improvement_Opt6: replaced copy.deepcopy(card) with model_copy().
        # copy.deepcopy() serialises and deserialises the entire Pydantic object graph
        # (including nested models) which is slow and GC-heavy.
        # model_copy() is Pydantic-native: it copies only the fields we mutate (capabilities)
        # and shares everything else by reference — much cheaper.
        # Old code: card = copy.deepcopy(card)
        from a2a.types import AgentCapabilities
        card = card.model_copy(update={"capabilities": AgentCapabilities(streaming=False)})
        logger.debug("[Lungo_Improvement_Opt6] Used model_copy for farm card: %s", farm)

        # Lungo_Improvement_Opt5: reuse cached A2A client for this farm.
        # a2a_client_factory.create() opens a SLIM/NATS session — expensive per-call.
        # We cache the client keyed by (farm_url, preferred_transport) so the session
        # is established once and reused for all subsequent requests to the same farm.
        transport_key = getattr(card, "preferred_transport", DEFAULT_MESSAGE_TRANSPORT)
        client_cache_key = (farm, str(transport_key))
        if client_cache_key not in _a2a_client_cache:
            logger.info("[Lungo_Improvement_Opt5] Creating new A2A client for farm=%s", farm)
            _a2a_client_cache[client_cache_key] = await a2a_client_factory.create(card)
        else:
            logger.info("[Lungo_Improvement_Opt5] Reusing cached A2A client for farm=%s", farm)
        client = _a2a_client_cache[client_cache_key]

        message = Message(
            messageId=str(uuid4()),
            role=Role.user,
            parts=[Part(TextPart(text=prompt))],
        )

        # Lungo_Improvement_Opt8: gate A2A calls through a semaphore.
        # Without a limit, concurrent requests could open unlimited simultaneous SLIM sessions,
        # overwhelming the transport layer. Semaphore(5) caps concurrent in-flight A2A calls.
        t0 = time.perf_counter()
        async with _a2a_semaphore:
            logger.debug("[Lungo_Improvement_Opt8] Acquired semaphore for farm=%s", farm)
            try:
                events = await send_a2a_with_retry(client, message)
            except Exception as session_exc:
                # Lungo_Improvement_Opt5 (stale-client eviction): if A2A fails with a
                # session error, the cached client has a dead SLIM session.  Evict it so the
                # next call creates a fresh client instead of re-using the broken one.
                if "session" in str(session_exc).lower() or "handshake" in str(session_exc).lower():
                    if client_cache_key in _a2a_client_cache:
                        del _a2a_client_cache[client_cache_key]
                        logger.warning(
                            "[Lungo_Improvement_Opt5] Evicted stale A2A client for farm=%s (cause: %s)",
                            farm, session_exc,
                        )
                        # Retry once with a fresh client
                        fresh_client = await a2a_client_factory.create(card)
                        _a2a_client_cache[client_cache_key] = fresh_client
                        logger.info("[Lungo_Improvement_Opt5] Reconnected A2A client for farm=%s", farm)
                        events = await send_a2a_with_retry(fresh_client, message)
                    else:
                        raise
                else:
                    raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[Lungo_Improvement] A2A round-trip for farm=%s took %.1f ms", farm, elapsed_ms)

        result_text = _extract_text_from_events(events)

        if result_text:
            # Lungo_Improvement_Opt4: store successful result in TTL cache
            _yield_cache[cache_key] = result_text
            logger.info("[Lungo_Improvement_Opt4] Cached result for farm=%s (cache size=%d)", farm, len(_yield_cache))
            return result_text
        else:
            raise A2AAgentError(f"Farm '{farm}' returned no text content.")
    except (TransportTimeoutError, RemoteAgentNoResponseError) as e:
        msg = "timed out" if isinstance(e, TransportTimeoutError) else "returned no response"
        logger.error(f"Failed to communicate with farm '{farm}': {msg}")
        raise A2AAgentError(f"Failed to communicate with farm '{farm}': {msg}.") from e
    except Exception as e: # Catch any underlying communication or client creation errors
        logger.error(f"Failed to communicate with farm '{farm}': {e}")
        raise A2AAgentError(f"Failed to communicate with farm '{farm}'. Details: {e}")

# node utility for streaming
async def get_all_farms_yield_inventory(prompt: str) -> str:
    """
    Broadcasts a prompt to all farms and aggregates their inventory responses.

    Args:
        prompt (str): The prompt to broadcast to all farm agents.

    Returns:
        str: A summary string containing yield information from all farms.
    """
    logger.info("entering get_all_farms_yield_inventory tool with prompt: %s", prompt)

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message=Message(
                messageId=str(uuid4()),
                role=Role.user,
                parts=[Part(TextPart(text=prompt))],
            ),
        )
    )

    # create a list of recipients to include in the broadcast
    recipients = [
        get_agent_identifier(get_farm_card(farm)) for farm in farm_registry
    ]

    try:
        # pick any card to initialize the client, will use the recipient list to route to the correct farms
        # Lungo_Improvement_Opt6: replaced copy.deepcopy() with model_copy() — Pydantic-native shallow copy.
        # copy.deepcopy() walks the entire object graph; model_copy() only copies what we override.
        # Old code: card = copy.deepcopy(farm_registry.cards()[0])
        card = farm_registry.cards()[0].model_copy(update={"preferred_transport": DEFAULT_MESSAGE_TRANSPORT.lower()})
        logger.debug("[Lungo_Improvement_Opt6] Used model_copy for broadcast card")

        # override preferred transport to ensure we use the intended publish-subscribe transport for broadcasts
        # Lungo_Improvement_Opt6: transport already set via model_copy above — removed redundant assignment
        # card.preferred_transport = DEFAULT_MESSAGE_TRANSPORT.lower()
        client = await a2a_client_factory.create(card)

        # create a broadcast message and collect responses
        responses = await client.broadcast_message(request, recipients=recipients)

        logger.info(f"got {len(responses)} responses back from farms")

        farm_yields = ""
        for response in responses:
            err = getattr(response.root, "error", None)
            result = getattr(response.root, "result", None)
            if err:
                err_msg = f"A2A error from farm: {err.message}"
                logger.error(err_msg)
                raise A2AAgentError(err_msg)
            if result and result.parts:
                part = result.parts[0].root
                farm_name = "Unknown Farm"
                if hasattr(result, "metadata") and result.metadata:
                    farm_name = result.metadata.get("name", "Unknown Farm")
                farm_yields += f"{farm_name} : {part.text.strip()}\n"
            else:
                err_msg = "Unknown response type from farm"
                logger.error(err_msg)
                raise A2AAgentError(err_msg)

        logger.info(f"Farm yields: {farm_yields}")
        return farm_yields.strip()
    except Exception as e: # Catch any underlying communication or client creation errors
        logger.error(f"Failed to communicate with all farms during broadcast: {e}")
        raise A2AAgentError(f"Failed to communicate with all farms. Details: {e}")

# node utility for streaming
async def get_all_farms_yield_inventory_streaming(prompt: str):
    """
    Broadcasts a prompt to all farms and streams their inventory responses as they arrive.

    Args:
        prompt (str): The prompt to broadcast to all farm agents.

    Yields:
        str: Yield information from each farm as it becomes available.
    """
    logger.info("entering get_all_farms_yield_inventory_streaming tool with prompt: %s", prompt)

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message=Message(
                messageId=str(uuid4()),
                role=Role.user,
                parts=[Part(TextPart(text=prompt))],
            ),
        )
    )

    # create a list of recipients to include in the broadcast
    recipients = [
        get_agent_identifier(get_farm_card(farm)) for farm in farm_registry
    ]

    try:
        logger.info(f"Broadcasting to {len(recipients)} farms: {', '.join(recipients)}")

        # Lungo_Improvement_Opt6: replaced copy.deepcopy() with model_copy() for streaming broadcast.
        # Same reasoning as non-streaming path: model_copy is faster and GC-cheaper than deepcopy.
        # Old code: card = copy.deepcopy(farm_registry.cards()[0])
        card = farm_registry.cards()[0].model_copy(update={"preferred_transport": DEFAULT_MESSAGE_TRANSPORT.lower()})
        logger.debug("[Lungo_Improvement_Opt6] Used model_copy for streaming broadcast card")

        # override preferred transport to ensure we use the intended publish-subscribe transport for broadcasts
        # Lungo_Improvement_Opt6: transport already set via model_copy above — removed redundant assignment
        # card.preferred_transport = DEFAULT_MESSAGE_TRANSPORT.lower()
        client = await a2a_client_factory.create(card)

        # Get the async generator for streaming responses
        response_stream = client.broadcast_message_streaming(
            request,
            recipients=recipients,
        )

        # Track which farms responded
        responded_farms = set()
        errors = []

        # Process responses as they arrive
        async for response in response_stream:
            try:
                err = getattr(response.root, "error", None)
                result = getattr(response.root, "result", None)
                if err:
                    err_msg = f"A2A error from farm: {err.message}"
                    logger.error(err_msg)
                    yield f"Error from farm: {err.message}\n"
                elif result and result.parts:
                    part = result.parts[0].root
                    farm_name = "Unknown Farm"
                    if hasattr(result, "metadata"):
                        farm_name = result.metadata.get("name", "Unknown Farm")

                    if farm_name == "None":
                        errors.append(part.text.strip())
                    else:
                        responded_farms.add(farm_name)
                        logger.info(f"Received response from {farm_name} ({len(responded_farms)}/{len(recipients)})")
                        yield f"{farm_name} : {part.text.strip()}\n"
                else:
                    err_msg = "Unknown response type from farm"
                    logger.error(err_msg)
                    yield f"Error: Unknown response format from farm\n"
            except Exception as e:
                logger.error(f"Error processing farm response: {e}")
                yield f"Error processing farm response: {str(e)}\n"

        # Check for missing responses and report them
        if len(responded_farms) < len(recipients):
            # Determine which farms didn't respond by checking farm names
            expected_farms = farm_registry.display_names()
            missing_farms = expected_farms - responded_farms

            if missing_farms:
                missing_list = ", ".join(sorted(missing_farms))
                logger.warning(f"Broadcast completed with partial responses: {len(responded_farms)}/{len(recipients)} farms responded. Missing: {missing_list}")

                response = f"No response from {missing_list}. These farms may be unavailable or slow to respond."
                if len(errors) != 0:
                    readable_errors = "\n".join(errors)
                    response += f" Errors encountered from farms:\n{readable_errors}\n"

                yield response


    except Exception as e:
        error_msg = f"Failed to communicate with farms during broadcast: {e}"
        logger.error(error_msg)
        # Check if it's a timeout-related error
        if "timeout" in str(e).lower():
            yield f"Error: Broadcast timed out. Some farms may be slow to respond or unavailable. {str(e)}\n"
        else:
            yield f"Error: {error_msg}\n"

@tool(args_schema=CreateOrderArgs)
@ioa_tool_decorator(name="create_order")
async def create_order(farm: str, quantity: int, price: float) -> str:
    """
    Sends a request to create a coffee order with a specific farm.

    Args:
        farm (str): The target farm for the order.
        quantity (int): Quantity of coffee to order.
        price (float): Proposed price per unit.

    Returns:
        str: Confirmation message or error string from the farm agent.

    Raises:
        A2AAgentError: If there's an issue with farm identification, identity verification, communication, or the farm agent returns an error.
        ValueError: For invalid input arguments.
    """

    farm = farm.strip().lower()

    logger.info(f"Creating order with price: {price}, quantity: {quantity}")
    if price <= 0 or quantity <= 0:
        raise ValueError("Price and quantity must be greater than zero.")

    if not farm:
        raise ValueError("No farm was provided, please provide a farm to create an order.")

    card = get_farm_card(farm)
    if card is None:
        raise ValueError(f"Farm '{farm}' not recognized. Available farms are: {', '.join(farm_registry.slugs())}.")

    logger.info(f"Using farm card: {card.name} for order creation")
    identity_service = IdentityServiceImpl(api_key=IDENTITY_API_KEY, base_url=IDENTITY_API_SERVER_URL)
    try:
        verify_farm_identity(identity_service, card.name)
    except Exception as e:
        # log the error and re-raise the exception
        raise A2AAgentError(f"Identity verification failed for farm '{farm}'. Details: {e}")

    try:
        # Lungo_Improvement_Opt6: replaced copy.deepcopy(card) with model_copy() for order creation.
        # model_copy() is Pydantic-native — no recursive serialisation, much faster than deepcopy.
        # Old code: card = copy.deepcopy(card)
        from a2a.types import AgentCapabilities
        card = card.model_copy(update={"capabilities": AgentCapabilities(streaming=False)})
        logger.debug("[Lungo_Improvement_Opt6] Used model_copy for create_order card: %s", farm)

        # Workaround: ioa-observe-sdk instruments SRPCTransport.send_message_streaming
        # with a coroutine wrapper (return await) instead of an async generator
        # (async for/yield), causing "TypeError: 'coroutine' object is not an
        # async iterator". Force the non-streaming path until fixed upstream.
        # See: https://github.com/agntcy/observe/issues/114
        # Lungo_Improvement_Opt6: streaming=False is now set inline via model_copy above.
        # if card.capabilities is None:
        #     from a2a.types import AgentCapabilities
        #     card.capabilities = AgentCapabilities(streaming=False)
        # else:
        #     card.capabilities.streaming = False

        client = await a2a_client_factory.create(card)

        message = Message(
            messageId=str(uuid4()),
            role=Role.user,
            parts=[Part(TextPart(text=f"Create an order with price {price} and quantity {quantity}"))],
        )

        events = await send_a2a_with_retry(client, message)
        result_text = _extract_text_from_events(events)

        if result_text:
            return result_text
        else:
            raise A2AAgentError(f"Farm '{farm}' returned no text content for order creation.")
    except (TransportTimeoutError, RemoteAgentNoResponseError) as e:
        msg = "timed out" if isinstance(e, TransportTimeoutError) else "returned no response"
        logger.error(f"Failed to communicate with order agent for farm '{farm}': {msg}")
        raise A2AAgentError(f"Failed to communicate with order agent for farm '{farm}': {msg}.") from e
    except Exception as e: # Catch any underlying communication or client creation errors
        logger.error(f"Failed to communicate with order agent for farm '{farm}': {e}")
        raise A2AAgentError(f"Failed to communicate with order agent for farm '{farm}'. Details: {e}")

@tool
@ioa_tool_decorator(name="get_order_details")
async def get_order_details(order_id: str) -> str:
    """
    Get details of an order.

    Args:
    order_id (str): The ID of the order.

    Returns:
    str: Details of the order.

    Raises:
    A2AAgentError: If there's an issue with communication or the order agent returns an error.
    ValueError: For invalid input arguments.
    """
    logger.info(f"Getting details for order ID: {order_id}")
    if not order_id:
        raise ValueError("Order ID must be provided.")

    try:
        # pick any card to initialize the client
        # Lungo_Improvement_Opt6: replaced copy.deepcopy() with model_copy() for get_order_details.
        # Sets preferred_transport and capabilities.streaming=False in one atomic model_copy call.
        # Old code: card = copy.deepcopy(farm_registry.cards()[0])
        from a2a.types import AgentCapabilities
        card = farm_registry.cards()[0].model_copy(update={
            "preferred_transport": InterfaceTransport.SLIM_RPC,
            "capabilities": AgentCapabilities(streaming=False),
        })
        logger.debug("[Lungo_Improvement_Opt6] Used model_copy for get_order_details card")

        # Workaround: ioa-observe-sdk instruments SRPCTransport.send_message_streaming
        # with a coroutine wrapper instead of an async generator, causing TypeError.
        # Lungo_Improvement_Opt6: now handled inline via model_copy above.
        # if card.capabilities is None: ...
        # card.capabilities.streaming = False

        client = await a2a_client_factory.create(card)

        message = Message(
            messageId=str(uuid4()),
            role=Role.user,
            parts=[Part(TextPart(text=f"Get details for order ID {order_id}"))],
        )

        events = await send_a2a_with_retry(client, message)
        result_text = _extract_text_from_events(events)

        if result_text:
            return result_text
        else:
            raise A2AAgentError(f"Order agent returned no text content for order ID '{order_id}'.")
    except (TransportTimeoutError, RemoteAgentNoResponseError) as e:
        msg = "timed out" if isinstance(e, TransportTimeoutError) else "returned no response"
        logger.error(f"Failed to communicate with order agent for order ID '{order_id}': {msg}")
        raise A2AAgentError(f"Failed to communicate with order agent for order ID '{order_id}': {msg}.") from e
    except Exception as e: # Catch any underlying communication or client creation errors
        logger.error(f"Failed to communicate with order agent for order ID '{order_id}': {e}")
        raise A2AAgentError(f"Failed to communicate with order agent for order ID '{order_id}'. Details: {e}")
