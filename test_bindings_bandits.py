"""Root-level smoke tests for bandit bindings.

Run:
  PYTHONPATH="$PWD/python" python3 test_bindings_bandits.py
"""

from ctorch import MAB, UCB


def test_mab() -> None:
    n_arms = 3
    mab = MAB(n_arms=n_arms, eps=0.2)

    arm = mab.select_arm()
    assert 0 <= arm < n_arms

    mab.update(arm, 1.0)
    mab.set_epsilon(0.1)

    arm_2 = mab.select_arm()
    assert 0 <= arm_2 < n_arms


def test_ucb() -> None:
    n_arms = 3
    ucb = UCB(n_arms=n_arms)

    for reward in (1.0, 0.0, 0.5, 0.75):
        arm = ucb.select_arm()
        assert 0 <= arm < n_arms
        ucb.update(arm, reward)


def main() -> None:
    test_mab()
    test_ucb()
    print("test_bindings_bandits.py: PASS")


if __name__ == "__main__":
    main()
