"""Video Poker single-turn environment built on the PrimeIntellect verifiers framework."""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Sequence

import verifiers as vf
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, State

RANK_SYMBOLS: tuple[str, ...] = (
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "J",
    "Q",
    "K",
    "A",
)
SUIT_SYMBOLS: tuple[str, ...] = ("♠", "♥", "♦", "♣")
FULL_DECK: tuple[str, ...] = tuple(
    f"{rank}{suit}" for suit in SUIT_SYMBOLS for rank in RANK_SYMBOLS
)
RANK_TO_VALUE: dict[str, int] = {rank: idx for idx, rank in enumerate(RANK_SYMBOLS)}
ROYAL_RANKS = frozenset({"T", "J", "Q", "K", "A"})

DEFAULT_PAYTABLE: dict[str, int] = {
    "royal_flush": 800,
    "straight_flush": 50,
    "four_of_a_kind": 25,
    "full_house": 9,
    "flush": 6,
    "straight": 4,
    "three_of_a_kind": 3,
    "two_pair": 2,
    "jacks_or_better": 1,
    "high_card": 0,
}
PAYTABLE_DISPLAY_NAMES: dict[str, str] = {
    "royal_flush": "Royal Flush",
    "straight_flush": "Straight Flush",
    "four_of_a_kind": "Four of a Kind",
    "full_house": "Full House",
    "flush": "Flush",
    "straight": "Straight",
    "three_of_a_kind": "Three of a Kind",
    "two_pair": "Two Pair",
    "jacks_or_better": "Jacks or Better (pair of Jacks, Queens, Kings, or Aces)",
    "high_card": "No Win",
}
PAYTABLE_ORDER: tuple[str, ...] = (
    "royal_flush",
    "straight_flush",
    "four_of_a_kind",
    "full_house",
    "flush",
    "straight",
    "three_of_a_kind",
    "two_pair",
    "jacks_or_better",
    "high_card",
)
DEFAULT_NUM_HANDS = 200
DEFAULT_PROMPT_STYLE = "standard"
ALT_PROMPT_STYLE = "dsl"

# Penalty configuration: encourage correct formatting and concise outputs by
# subtracting from the reward when completions violate requirements. Correct
# behaviour does not earn bonus reward.
FORMAT_MISS_PENALTY = 0.5
LONG_COMPLETION_CHAR_THRESHOLD = 2048
LONG_COMPLETION_PENALTY = 0.2

INDEX_PATTERN = re.compile(r"[0-4]")
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


class VideoPokerEnv(SingleTurnEnv):
    """Single-turn video poker environment."""

    def __init__(
        self,
        *,
        paytable: dict[str, int] | None = None,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        rubric: Rubric | None = None,
        **kwargs,
    ) -> None:
        self.paytable = dict(paytable or DEFAULT_PAYTABLE)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            **kwargs,
        )

    async def setup_state(self, state: State, **kwargs) -> State:  # type: ignore[override]
        info = state.get("info")
        if not isinstance(info, dict):
            info = {}
            state["info"] = info
        if "paytable" not in info:
            info["paytable"] = dict(self.paytable)
        return await super().setup_state(state, **kwargs)


def build_video_poker_dataset(
    *,
    num_hands: int = DEFAULT_NUM_HANDS,
    seed: int | None = None,
    paytable: dict[str, int] | None = None,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> Dataset:
    """Create a dataset of random single-turn video poker prompts."""

    if num_hands <= 0:
        raise ValueError("num_hands must be positive")
    rng = random.Random(seed)
    active_paytable = dict(paytable or DEFAULT_PAYTABLE)
    prompts = []
    for _ in range(num_hands):
        hand = tuple(rng.sample(FULL_DECK, 5))
        prompts.append(
            {
                "prompt": [
                    _format_prompt(hand, active_paytable, prompt_style)
                ],
                "answer": "",
                "task": "video_poker",
                "info": {
                    "initial_hand": list(hand),
                    "paytable": dict(active_paytable),
                },
            }
        )
    return Dataset.from_list(prompts)


def _format_prompt(
    hand: Sequence[str], paytable: dict[str, int], prompt_style: str
) -> dict[str, str]:
    hand_lines = "\n".join(f"{idx}: {card}" for idx, card in enumerate(hand))
    paytable_lines = "\n".join(
        f"{PAYTABLE_DISPLAY_NAMES.get(key, key.replace('_', ' ').title())}: {paytable[key]}"
        for key in PAYTABLE_ORDER
        if key in paytable
    )
    if prompt_style == ALT_PROMPT_STYLE:
        content = (
            "You are playing a single-hand game of Jacks or Better video poker.\n"
            "Your current five-card hand (index: card) is:\n"
            f"{hand_lines}\n\n"
            "Payout table (per 1 credit wagered):\n"
            f"{paytable_lines}\n\n"
            "Inside <think> use PokerDSL v1 EXACTLY as specified below, one line per field, no extra words, ≤100 tokens:\n\n"
            "H: <5 cards>\n"
            "RC: <rank counts as (R×n)...>\n"
            "SC: <suit counts as (♠×a)(♥×b)(♦×c)(♣×d)>\n"
            "C: <up to 5 candidate holds as [i ...] ... always include pairs/3K/2P/4F/4SF/4RF/3RF/OESD/ISD>\n"
            "S: <scores as {[i ...]: tier,outs} corresponding to C; tier ∈ [PAT,4RF,4SF,3K,2P,HP,4F,LP,OESD,ISD,HC]>\n"
            "B: <best indices only>\n\n"
            "After </think>, output <answer>HOLD: i j k</answer> using ascending zero-based indices.\n"
            "Leave the indices empty (i.e., `<answer>HOLD:</answer>`) if discarding all cards.\n"
            "No other text outside the required tags."
        )
    else:
        content = (
            "You are playing a single-hand game of Jacks or Better video poker.\n"
            "Your current five-card hand (index: card) is:\n"
            f"{hand_lines}\n\n"
            "Payout table (per 1 credit wagered):\n"
            f"{paytable_lines}\n\n"
            "Think step-by-step about which cards to hold inside <think>...</think> tags.\n"
            "After thinking, provide your final decision inside <answer>...</answer> tags.\n"
            "Format the content of <answer> exactly as `HOLD: i j k` with indices in ascending order.\n"
            "Use zero-based indices. If you will discard all cards, respond with `<answer>HOLD:</answer>`.\n"
            "Do not include any other text outside the required tags. The environment will compute the expected value of your decision."
        )
    return {"role": "user", "content": content}


def _merge_completion_text(completion: Messages) -> str:
    if isinstance(completion, str):
        return completion
    texts: list[str] = []
    for message in completion:
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, Iterable):  # tool/text message chunks
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    chunk_text = chunk.get("text", "")
                    if chunk_text:
                        texts.append(str(chunk_text))
    return "\n".join(texts)


def _extract_answer_text(full_text: str) -> str:
    text = full_text.strip()
    answer_match = ANSWER_TAG_PATTERN.search(text)
    if answer_match:
        return answer_match.group(1).strip()
    return text


def extract_completion_text(completion: Messages) -> str:
    return _extract_answer_text(_merge_completion_text(completion))


def extract_think_texts(completion: Messages | str) -> list[str]:
    full_text = completion if isinstance(completion, str) else _merge_completion_text(completion)
    return [segment.strip() for segment in THINK_TAG_PATTERN.findall(full_text)]


def parse_hold_indices(text: str, hand_size: int = 5) -> list[int]:
    raw_indices = {int(match) for match in INDEX_PATTERN.findall(text)}
    return sorted(index for index in raw_indices if 0 <= index < hand_size)


def video_poker_reward(
    *,
    prompt: Messages,
    completion: Messages,
    answer: str,
    state: State,
    info: Info | None = None,
    **_: object,
) -> float:
    del prompt, answer
    merged_info: dict[str, object] = {}
    state_info = state.get("info")
    if isinstance(state_info, dict):
        merged_info.update(state_info)
    if isinstance(info, dict):
        merged_info.update(info)
    hand = merged_info.get("initial_hand")
    paytable_raw = merged_info.get("paytable", DEFAULT_PAYTABLE)
    if isinstance(hand, (list, tuple)):
        hand_cards = list(hand)
    else:
        return 0.0
    paytable = _normalize_paytable(paytable_raw)
    completion_text = _merge_completion_text(completion)
    action_text = _extract_answer_text(completion_text)
    hold_indices = parse_hold_indices(action_text, hand_size=len(hand_cards))
    think_segments = extract_think_texts(completion_text)

    penalties: dict[str, float] = {
        "format": 0.0,
        "length": 0.0,
    }
    if not ANSWER_TAG_PATTERN.search(completion_text):
        penalties["format"] = -FORMAT_MISS_PENALTY
    if len(completion_text) > LONG_COMPLETION_CHAR_THRESHOLD:
        penalties["length"] = -LONG_COMPLETION_PENALTY

    expected_value = compute_expected_value(
        tuple(hand_cards), tuple(hold_indices), paytable
    )
    total_reward = expected_value + sum(penalties.values())
    held_cards = [hand_cards[idx] for idx in hold_indices]
    state["parsed_action"] = {
        "text": action_text,
        "hold_indices": hold_indices,
        "held_cards": held_cards,
        "think": think_segments,
    }
    state["raw_completion_text"] = completion_text.strip()
    state["expected_value"] = expected_value
    state["reward_penalties"] = penalties
    state["total_reward"] = total_reward
    return total_reward


def compute_expected_value(
    hand: Sequence[str],
    hold_indices: Sequence[int],
    paytable: dict[str, int] | None = None,
) -> float:
    paytable_key = tuple(sorted((k, int(v)) for k, v in (paytable or DEFAULT_PAYTABLE).items()))
    hand_key = tuple(hand)
    hold_key = tuple(sorted({idx for idx in hold_indices if 0 <= idx < len(hand)}))
    return _expected_value_cached(hand_key, hold_key, paytable_key)


@lru_cache(maxsize=65536)
def _expected_value_cached(
    hand_key: tuple[str, ...],
    hold_key: tuple[int, ...],
    paytable_key: tuple[tuple[str, int], ...],
) -> float:
    paytable = {k: v for k, v in paytable_key}
    hand = list(hand_key)
    hold_indices = list(hold_key)
    hold_cards = [hand[idx] for idx in hold_indices]
    draw_needed = len(hand) - len(hold_cards)
    if draw_needed == 0:
        return float(evaluate_hand(hold_cards, paytable))
    hand_set = set(hand)
    remaining_deck = [card for card in FULL_DECK if card not in hand_set]
    total = 0.0
    count = 0
    for draw in itertools.combinations(remaining_deck, draw_needed):
        final_hand = hold_cards + list(draw)
        total += evaluate_hand(final_hand, paytable)
        count += 1
    return total / count if count else 0.0


def evaluate_hand(hand: Sequence[str], paytable: dict[str, int]) -> int:
    ranks = [card[0] for card in hand]
    suits = [card[-1] for card in hand]
    is_flush = len(set(suits)) == 1
    rank_counts: dict[str, int] = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    count_values = sorted(rank_counts.values(), reverse=True)
    values = sorted(RANK_TO_VALUE[rank] for rank in ranks)
    is_straight = _is_straight(values)
    if is_flush and is_straight:
        if set(ranks) == ROYAL_RANKS:
            return paytable.get("royal_flush", 0)
        return paytable.get("straight_flush", 0)
    if 4 in count_values:
        return paytable.get("four_of_a_kind", 0)
    if count_values == [3, 2]:
        return paytable.get("full_house", 0)
    if is_flush:
        return paytable.get("flush", 0)
    if is_straight:
        return paytable.get("straight", 0)
    if 3 in count_values:
        return paytable.get("three_of_a_kind", 0)
    if count_values.count(2) == 2:
        return paytable.get("two_pair", 0)
    if 2 in count_values:
        for rank, count in rank_counts.items():
            if count == 2 and RANK_TO_VALUE[rank] >= RANK_TO_VALUE["J"]:
                return paytable.get("jacks_or_better", 0)
    return paytable.get("high_card", 0)


def _is_straight(values: Sequence[int]) -> bool:
    unique = sorted(set(values))
    if len(unique) != 5:
        return False
    # Handle wheel straight A-2-3-4-5
    if unique == [0, 1, 2, 3, RANK_TO_VALUE["A"]]:
        return True
    return unique[-1] - unique[0] == 4


def _normalize_paytable(paytable: object) -> dict[str, int]:
    if isinstance(paytable, dict):
        return {str(k): int(v) for k, v in paytable.items()}
    if isinstance(paytable, list):
        return {str(k): int(v) for k, v in paytable}
    return dict(DEFAULT_PAYTABLE)


def load_environment(
    *,
    paytable: dict[str, int] | None = None,
    dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    rubric: Rubric | None = None,
    num_hands: int = DEFAULT_NUM_HANDS,
    seed: int | None = None,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    **kwargs,
) -> vf.Environment:
    """Instantiate the video poker environment."""

    paytable_dict = dict(paytable or DEFAULT_PAYTABLE)
    if dataset is None:
        dataset = build_video_poker_dataset(
            num_hands=num_hands,
            seed=seed,
            paytable=paytable_dict,
            prompt_style=prompt_style,
        )
    if rubric is None:
        rubric = Rubric(
            funcs=[video_poker_reward],
            weights=[1.0],
            parallelize_scoring=False,
        )
    env = VideoPokerEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        paytable=paytable_dict,
        **kwargs,
    )
    return env


__all__ = [
    "VideoPokerEnv",
    "build_video_poker_dataset",
    "compute_expected_value",
    "evaluate_hand",
    "load_environment",
]
