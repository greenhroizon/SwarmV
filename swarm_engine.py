"""Deterministic, dependency-free heuristic swarm forecasting.

This module deliberately does not call a model, a network service, or a random
number generator.  It creates a repeatable illustrative panel from the input
text and reports the assumptions that went into the result.  The output is a
structured forecast, not a claim about the future or a live multi-agent run.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


SUPPORTED_SWARM_SIZES = (24, 128, 512, 1000)
MIN_SWARM_SIZE = 24
MAX_SWARM_SIZE = 5_000
DEFAULT_SWARM_SIZE = 128
DEFAULT_DEBATE_ROUNDS = 2
MIN_DEBATE_ROUNDS = 1
MAX_DEBATE_ROUNDS = 5
MAX_SCENARIO_LENGTH = 2_000
MAX_CONTEXT_LENGTH = 1_200
MAX_DOMAINS = 8
MAX_DOMAIN_LENGTH = 48
REPRESENTATIVE_ROSTER_SIZE = 8

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")

# These are intentionally small directional cues.  They are not a language
# model and should not be interpreted as evidence about the real world.
_POSITIVE_CUES = {
    "approved": 0.10,
    "backed": 0.08,
    "certain": 0.10,
    "confirmed": 0.11,
    "growth": 0.07,
    "likely": 0.11,
    "momentum": 0.07,
    "on-track": 0.08,
    "ready": 0.07,
    "success": 0.09,
    "supported": 0.08,
    "win": 0.07,
}
_NEGATIVE_CUES = {
    "blocked": -0.11,
    "cancelled": -0.12,
    "decline": -0.08,
    "denied": -0.11,
    "delay": -0.08,
    "fail": -0.10,
    "failed": -0.11,
    "fragile": -0.07,
    "loss": -0.08,
    "risk": -0.06,
    "shortage": -0.10,
    "uncertain": -0.05,
    "unlikely": -0.12,
}

_ROLE_TEMPLATES = (
    "base-rate analyst",
    "domain specialist",
    "risk reviewer",
    "scenario planner",
    "evidence auditor",
    "operations observer",
    "contrarian reviewer",
    "synthesis lead",
)


class ValidationError(ValueError):
    """A user-correctable simulation input error.

    ``field`` is suitable for an API error response and is optional for
    request-level errors.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field


def _first_present(
    payload: Mapping[str, Any],
    canonical: str,
    aliases: Sequence[str],
    default: Any = None,
) -> Any:
    """Read a canonical JSON key while accepting the frontend-style alias."""

    names = (canonical, *aliases)
    present = [(name, payload[name]) for name in names if name in payload]
    if not present:
        return default

    first_name, first_value = present[0]
    for name, value in present[1:]:
        if value != first_value:
            raise ValidationError(
                f"provide only one value for {canonical} (conflicting {first_name} and {name})",
                canonical,
            )
    return first_value


def _validate_scenario(value: Any) -> str:
    if not isinstance(value, str):
        raise ValidationError("scenario must be a string", "scenario")
    scenario = value.strip()
    if not scenario:
        raise ValidationError("scenario must not be empty", "scenario")
    if len(scenario) > MAX_SCENARIO_LENGTH:
        raise ValidationError(
            f"scenario must be at most {MAX_SCENARIO_LENGTH} characters", "scenario"
        )
    return scenario


def _validate_context(value: Any) -> str:
    """Normalize optional user-supplied conditions without pretending to verify them."""

    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValidationError("context must be a string", "context")
    context = value.strip()
    if len(context) > MAX_CONTEXT_LENGTH:
        raise ValidationError(
            f"context must be at most {MAX_CONTEXT_LENGTH} characters", "context"
        )
    return context


def _validate_swarm_size(value: Any) -> int:
    # bool is an int subclass, but it is not a meaningful swarm size.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"swarm_size must be an integer from {MIN_SWARM_SIZE} to {MAX_SWARM_SIZE}",
            "swarm_size",
        )
    if not MIN_SWARM_SIZE <= value <= MAX_SWARM_SIZE:
        raise ValidationError(
            f"swarm_size must be an integer from {MIN_SWARM_SIZE} to {MAX_SWARM_SIZE}",
            "swarm_size",
        )
    return value


def _validate_domains(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("domains must be a non-empty list of strings", "domains")
    if not value:
        raise ValidationError("domains must contain at least one domain", "domains")
    if len(value) > MAX_DOMAINS:
        raise ValidationError(f"domains may contain at most {MAX_DOMAINS} items", "domains")

    domains: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValidationError(f"domains[{index}] must be a string", "domains")
        domain = item.strip()
        if not domain:
            raise ValidationError(f"domains[{index}] must not be empty", "domains")
        if len(domain) > MAX_DOMAIN_LENGTH:
            raise ValidationError(
                f"domains[{index}] must be at most {MAX_DOMAIN_LENGTH} characters", "domains"
            )
        key = domain.casefold()
        if key in seen:
            raise ValidationError("domains must not contain duplicates", "domains")
        seen.add(key)
        domains.append(domain)
    return domains


def _validate_debate_rounds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"debate_rounds must be an integer from {MIN_DEBATE_ROUNDS} to {MAX_DEBATE_ROUNDS}",
            "debate_rounds",
        )
    if not MIN_DEBATE_ROUNDS <= value <= MAX_DEBATE_ROUNDS:
        raise ValidationError(
            f"debate_rounds must be an integer from {MIN_DEBATE_ROUNDS} to {MAX_DEBATE_ROUNDS}",
            "debate_rounds",
        )
    return value


def validate_simulation_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a simulation request.

    The canonical keys are ``scenario``, ``swarm_size``, ``domains``, and
    ``debate_rounds``.  ``swarmSize`` and ``debateRounds`` are accepted as
    harmless compatibility aliases for browser clients.
    """

    if not isinstance(payload, Mapping):
        raise ValidationError("request body must be a JSON object")

    # ``question`` is the user-facing name used by the live workflow.  Keep
    # ``scenario`` as the canonical response/API field for existing clients.
    scenario = _validate_scenario(_first_present(payload, "scenario", ("question",), None))
    context = _validate_context(payload.get("context", ""))
    swarm_size = _validate_swarm_size(
        _first_present(payload, "swarm_size", ("swarmSize",), DEFAULT_SWARM_SIZE)
    )
    domains = _validate_domains(payload.get("domains", ["general"]))
    debate_rounds = _validate_debate_rounds(
        _first_present(payload, "debate_rounds", ("debateRounds",), DEFAULT_DEBATE_ROUNDS)
    )
    return {
        "scenario": scenario,
        "context": context,
        "swarm_size": swarm_size,
        "domains": domains,
        "debate_rounds": debate_rounds,
    }


def _digest(*parts: object) -> bytes:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).digest()


def _unit_value(*parts: object) -> float:
    """Return a stable value in [0, 1) without using pseudo-random state."""

    digest = _digest(*parts)
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _signed_value(*parts: object) -> float:
    return (_unit_value(*parts) * 2.0) - 1.0


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _round(value: float, places: int = 3) -> float:
    return round(float(value), places)


def _tokens(scenario: str) -> list[str]:
    return _TOKEN_RE.findall(scenario.casefold())


def _cue_score(tokens: Sequence[str]) -> tuple[float, list[str], list[str]]:
    score = 0.0
    positive: list[str] = []
    negative: list[str] = []
    for token in tokens:
        if token in _POSITIVE_CUES:
            score += _POSITIVE_CUES[token]
            positive.append(token)
        if token in _NEGATIVE_CUES:
            score += _NEGATIVE_CUES[token]
            negative.append(token)
    return _clamp(score, -0.32, 0.32), positive, negative


def _domain_signal(scenario: str, domains: Sequence[str]) -> float:
    if not domains:
        return 0.0
    signals = [_signed_value("domain-signal", scenario.casefold(), domain.casefold()) for domain in domains]
    return sum(signals) / len(signals) * 0.045


def _agent_domain(domains: Sequence[str], index: int, scenario: str) -> str:
    # The hash changes which specialist sees the scenario first while the
    # modulo keeps the result easy to explain and fully deterministic.
    offset = int.from_bytes(_digest("domain-order", scenario, index)[:2], "big")
    return domains[(index + offset) % len(domains)]


def _initial_stance(
    scenario: str,
    domains: Sequence[str],
    index: int,
    prior: float,
) -> float:
    domain = _agent_domain(domains, index, scenario)
    # Independent-looking variation illustrates a panel without pretending
    # that these are independent real people or model calls.
    noise = _signed_value("agent-noise", scenario.casefold(), index) * 0.14
    expertise = _signed_value("agent-domain", scenario.casefold(), domain.casefold(), index) * 0.045
    return _clamp(prior + noise + expertise, 0.04, 0.96)


def _agent_label(stance: float) -> str:
    if stance >= 0.62:
        return "supportive"
    if stance <= 0.38:
        return "skeptical"
    return "mixed"


def _representative_roster(
    scenario: str,
    domains: Sequence[str],
    stances: Sequence[float],
) -> list[dict[str, Any]]:
    roster: list[dict[str, Any]] = []
    count = min(REPRESENTATIVE_ROSTER_SIZE, len(stances))
    for index in range(count):
        domain = _agent_domain(domains, index, scenario)
        stance = stances[index]
        role = _ROLE_TEMPLATES[index % len(_ROLE_TEMPLATES)]
        if role == "domain specialist":
            role = f"{domain} specialist"
        roster.append(
            {
                "id": f"agent-{index + 1:03d}",
                "role": role,
                "domain": domain,
                "stance": _agent_label(stance),
                "estimated_probability": _round(stance),
                "rationale": (
                    f"A deterministic {role} lens with a bounded synthetic stance; "
                    "it is a representative perspective, not a real participant."
                ),
            }
        )
    return roster


def _debate_messages(
    scenario: str,
    domains: Sequence[str],
    rounds: int,
    probability: float,
    dispersion: float,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for round_number in range(1, rounds + 1):
        challenge = _signed_value("debate", scenario.casefold(), round_number) * 0.04
        direction = "supporting" if challenge >= 0 else "challenging"
        domain = domains[(round_number - 1) % len(domains)]
        messages.append(
            {
                "round": round_number,
                "speaker": "synthesis lead",
                "message": (
                    f"Round {round_number}: the synthetic panel is {direction} the "
                    f"current {domain} view; center estimate {_round(probability * 100, 1)}%. "
                    f"Dispersion is {_round(dispersion, 3)}, so this remains an illustrative "
                    "heuristic rather than consensus evidence."
                ),
            }
        )
    return messages


def _driver_text(
    tokens: Sequence[str],
    positive_cues: Sequence[str],
    negative_cues: Sequence[str],
    domains: Sequence[str],
    swarm_size: int,
    debate_rounds: int,
    context: str,
) -> list[str]:
    drivers: list[str] = []
    if positive_cues or negative_cues:
        cues = [*positive_cues, *negative_cues]
        drivers.append("Directional wording detected: " + ", ".join(cues[:6]) + ".")
    else:
        drivers.append("No strong directional keywords were detected; the center stays near neutral.")
    drivers.append("Domain lenses included: " + ", ".join(domains) + ".")
    drivers.append(
        f"The estimate aggregates {swarm_size} deterministic synthetic stances across "
        f"{debate_rounds} debate round{'s' if debate_rounds != 1 else ''}."
    )
    if context:
        drivers.append("User-supplied conditions were included as directional input, but not independently verified.")
    if len(tokens) < 8:
        drivers.append("The short scenario provides limited textual signal to this heuristic.")
    return drivers


def _uncertainty_text(
    scenario: str,
    context: str,
    domains: Sequence[str],
    swarm_size: int,
    dispersion: float,
) -> list[str]:
    uncertainties = [
        "No live data, base rates, external sources, or real agents are consulted.",
        "Keyword cues and a stable hash provide illustrative variation; they do not establish causality.",
        f"The synthetic panel's stance dispersion is {_round(dispersion)}, so minority views may matter.",
        "A virtual swarm size changes aggregation stability, not the amount of real-world evidence.",
    ]
    if len(domains) == 1:
        uncertainties.append("A single domain lens may miss operational, social, or market context.")
    if context:
        uncertainties.append("The supplied conditions are assumptions, not verified observations or evidence.")
    if len(scenario) < 80:
        uncertainties.append("Additional timing, definitions, and outcome criteria could materially change the estimate.")
    # Keep the response compact and stable while retaining a useful caveat set.
    return uncertainties[:5]


def _simulate(request: Mapping[str, Any]) -> dict[str, Any]:
    scenario = request["scenario"]
    context = request["context"]
    swarm_size = request["swarm_size"]
    domains = request["domains"]
    debate_rounds = request["debate_rounds"]
    brief = "\n".join(part for part in (scenario, context) if part)
    tokens = _tokens(brief)
    cue_score, positive_cues, negative_cues = _cue_score(tokens)
    hash_signal = _signed_value("scenario-prior", brief.casefold(), "|".join(domains)) * 0.07
    prior = _clamp(0.5 + cue_score + hash_signal + _domain_signal(brief, domains), 0.08, 0.92)

    stances = [_initial_stance(brief, domains, index, prior) for index in range(swarm_size)]
    # Each round nudges the synthetic panel slightly toward its current mean.
    # This is a transparent consensus mechanic, not a simulated conversation.
    for round_number in range(debate_rounds):
        center = sum(stances) / len(stances)
        convergence = 0.06 + (round_number * 0.015)
        stances = [
            _clamp(
                stance + (center - stance) * convergence
                + _signed_value("reconsider", brief, index, round_number) * 0.006,
                0.04,
                0.96,
            )
            for index, stance in enumerate(stances)
        ]

    probability = _clamp(sum(stances) / len(stances), 0.02, 0.98)
    variance = sum((stance - probability) ** 2 for stance in stances) / len(stances)
    dispersion = math.sqrt(variance)
    consensus = _clamp(1.0 - dispersion * 3.5)
    text_signal = _clamp(len(tokens) / 24.0)
    sample_stability = _clamp(0.48 + (math.log2(swarm_size) - math.log2(24)) * 0.075, 0.48, 0.88)
    confidence = _clamp(
        0.20
        + consensus * 0.28
        + sample_stability * 0.20
        + text_signal * 0.17
        + min(debate_rounds, 4) * 0.025,
        0.18,
        0.90,
    )

    response = {
        "scenario": scenario,
        "context": context,
        "swarm_size": swarm_size,
        "domains": list(domains),
        "debate_rounds": debate_rounds,
        "probability": _round(probability),
        "probability_percent": _round(probability * 100, 1),
        "confidence": _round(confidence),
        "confidence_percent": _round(confidence * 100, 1),
        "outcome": (
            "The synthetic panel leans toward the scenario occurring"
            if probability >= 0.5
            else "The synthetic panel leans against the scenario occurring"
        )
        + f" at {probability * 100:.1f}% directional likelihood.",
        "key_drivers": _driver_text(
            tokens, positive_cues, negative_cues, domains, swarm_size, debate_rounds, context
        ),
        "uncertainties": _uncertainty_text(scenario, context, domains, swarm_size, dispersion),
        "representative_roster": _representative_roster(brief, domains, stances),
        "debate_messages": _debate_messages(
            brief, domains, debate_rounds, probability, dispersion
        ),
        "method": {
            "name": "deterministic-heuristic-swarm",
            "description": "A dependency-free illustrative aggregation of bounded synthetic stances.",
            "live_agents": False,
            "external_data": False,
            "context_used": bool(context),
        },
    }
    return response


def simulate_forecast(
    scenario_or_request: str | Mapping[str, Any],
    swarm_size: int | None = None,
    domains: Sequence[str] | None = None,
    debate_rounds: int | None = None,
) -> dict[str, Any]:
    """Return a deterministic forecast for a request or scenario string.

    Mapping input is the preferred API for servers.  The optional positional
    arguments make the engine convenient to use directly in a small script.
    """

    if isinstance(scenario_or_request, Mapping):
        if any(value is not None for value in (swarm_size, domains, debate_rounds)):
            raise ValidationError("do not combine a request object with separate simulation arguments")
        request = validate_simulation_request(scenario_or_request)
    else:
        request = validate_simulation_request(
            {
                "scenario": scenario_or_request,
                "swarm_size": DEFAULT_SWARM_SIZE if swarm_size is None else swarm_size,
                "domains": ["general"] if domains is None else domains,
                "debate_rounds": DEFAULT_DEBATE_ROUNDS if debate_rounds is None else debate_rounds,
            }
        )
    return _simulate(request)


# A descriptive alias for callers that prefer the noun used by the product.
run_simulation = simulate_forecast


__all__ = [
    "DEFAULT_DEBATE_ROUNDS",
    "DEFAULT_SWARM_SIZE",
    "MAX_DEBATE_ROUNDS",
    "MAX_SCENARIO_LENGTH",
    "MAX_CONTEXT_LENGTH",
    "MAX_SWARM_SIZE",
    "MIN_SWARM_SIZE",
    "SUPPORTED_SWARM_SIZES",
    "ValidationError",
    "run_simulation",
    "simulate_forecast",
    "validate_simulation_request",
]
