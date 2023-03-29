#include "lbfgsb-alm.hpp"
#include <alpaqa/implementation/outer/alm.tpp>

namespace alpaqa {

template class ALMSolver<lbfgsb::LBFGSBSolver>;

} // namespace alpaqa
