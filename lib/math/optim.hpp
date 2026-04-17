#pragma once

// Compatibility umbrella header. Existing includes of "math/optim.hpp"
// continue to work while implementations live in dedicated files.

#include "optim_base.hpp"
#include "optim_gd.hpp"
#include "optim_sgd.hpp"
#include "optim_adagrad.hpp"
#include "optim_rmsprop.hpp"
#include "optim_adam.hpp"
#include "optim_adamw.hpp"
#include "optimizer.hpp"

