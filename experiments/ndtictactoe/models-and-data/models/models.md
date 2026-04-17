z# Architectures

## Architecture 1

32 -> 128 -> ReLU -> 16

AI first

About 60% win rate

## Architecture 2

AI first or second

32 -> 256 -> ReLU -> 256 -> ReLU -> 128 -> ReLU -> 64 -> ReLU -> 16

Works really poorly because the reward was not configured properly


## Architecture 3

AI first

32 -> 128 -> ReLU -> 128 -> ReLU -> 16

About 95% win rate

# Architecture 4

AI first

18 -> 32 -> ReLU -> 32 -> 9

96% win at 20000 episodes

Kind of unstable