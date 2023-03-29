#pragma once

#include <alpaqa/lbfgsb/lbfgsb-adapter.hpp>
#include <alpaqa/outer/alm.hpp>

namespace alpaqa {

extern template class ALMSolver<lbfgsb::LBFGSBSolver>;

} // namespace alpaqa
