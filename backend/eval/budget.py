"""A hard spend cap for paid evaluation runs, on top of the egress guard (app/egress_guard.py).

`--max-usd X` installs the egress guard in the evaluation process with one approved host
(api.openai.com), the local stores as internal hosts, and X as its spend cap. The guard
adds every OpenAI request's worst case (each byte of the body counted as an input token,
the request's `max_tokens` as output) to a running bound BEFORE the request is sent, and
refuses the request if the bound would pass X. The real spend is always below that bound,
so it is below X: a hard cap, not an estimate.

A refused request is not a normal failure: the chat pipeline falls back when its plan or
verify call fails, which would quietly turn a cut-off question into a worse-looking
answer. So the runner checks the guard after every question (`check`) and stops the run
when it tripped; that question's result is thrown away, the ones before it are kept. To
stop cleanly in the usual case, it also stops BEFORE a question when the cap left is
smaller than the largest bound one question has added so far (`before_question`).
"""

from dataclasses import dataclass, field
from decimal import Decimal

from app.config import Settings
from app.egress_guard import Guard, Policy, install

PAID_HOST = "api.openai.com"
INTERNAL_HOSTS = ("localhost", "127.0.0.1", "::1", "postgres", "qdrant", "redis")


class BudgetExhausted(RuntimeError):
    """The spend cap stops the run (the current question is not stored)."""


def policy(max_usd: Decimal, max_requests: int, s: Settings) -> Policy:
    """One paid host and the local stores. The prices are the highest the run can be
    billed at (chat or judge model; the dearest embedding model a variant uses), so the
    bound never undercounts."""
    embedding = max(s.embedding_price_per_1m, s.eval_embedding_price_per_1m,
                    s.eval_large_embedding_price_per_1m)  # fmt: skip
    return Policy(
        hosts={PAID_HOST: max_requests},
        total=max_requests,
        internal_hosts=frozenset(INTERNAL_HOSTS),
        spend_cap_usd=max_usd,
        paid_host=PAID_HOST,
        prices_per_1m={
            "chat_input": Decimal(str(max(s.chat_price_in_per_1m, s.eval_price_in_per_1m))),
            "chat_output": Decimal(str(max(s.chat_price_out_per_1m, s.eval_price_out_per_1m))),
            "embedding": Decimal(str(embedding)),
        },
    )


@dataclass
class Budget:
    guard: Guard
    largest_question_usd: Decimal = Decimal("0")
    _start: Decimal = field(default=Decimal("0"), repr=False)

    @property
    def cap_usd(self) -> Decimal:
        return self.guard.policy.spend_cap_usd

    @property
    def bound_usd(self) -> Decimal:
        return self.guard.spend_bound_usd

    def before_question(self) -> None:
        """Stop before a question that could not be finished within the cap."""
        self.check()
        left = self.cap_usd - self.bound_usd
        if self.largest_question_usd and left < self.largest_question_usd:
            raise BudgetExhausted(
                f"spend cap ${self.cap_usd}: ${left:.6f} left, a question has needed up to "
                f"${self.largest_question_usd:.6f} (worst case)"
            )
        self._start = self.bound_usd

    def check(self) -> None:
        """Stop if the guard refused anything (then the current question is incomplete)."""
        if self.guard.tripped is not None:
            raise BudgetExhausted(f"the egress guard stopped the run: {self.guard.tripped}")

    def after_question(self) -> None:
        self.check()
        self.largest_question_usd = max(self.largest_question_usd, self.bound_usd - self._start)


def install_budget(max_usd: Decimal, s: Settings, *, max_requests: int = 100_000) -> Budget:
    """Install the guard for this process (once) and return the run's budget."""
    if max_usd <= 0:
        raise ValueError("--max-usd must be above 0")
    return Budget(install(Guard(policy(max_usd, max_requests, s))))
