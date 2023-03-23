#include "params.tpp"

#include <alpaqa/lbfgsb-adapter.hpp>

#include <stdexcept>

namespace alpaqa {

PARAMS_TABLE(lbfgspp::LBFGSBSolver<config_t>::Params, //
             PARAMS_MEMBER(m),                        //
             PARAMS_MEMBER(past),                     //
             PARAMS_MEMBER(delta),                    //
             PARAMS_MEMBER(max_iterations),           //
             PARAMS_MEMBER(max_submin),               //
             PARAMS_MEMBER(max_linesearch),           //
             PARAMS_MEMBER(min_step),                 //
             PARAMS_MEMBER(max_step),                 //
             PARAMS_MEMBER(ftol),                     //
             PARAMS_MEMBER(wolfe),                    //
);

template void set_params(lbfgspp::LBFGSBSolver<config_t>::Params &,
                         std::string_view, std::span<const std::string_view>);

} // namespace alpaqa