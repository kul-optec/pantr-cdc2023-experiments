TOL := 1e-8
DELTA := 1
OUTPUT := output

QUADCOPTER_RESULTS := \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-tr-newton_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-panoc200_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-panoc50_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-ipopt-precompiled_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-strucpanoc50_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-gn_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-panoc20_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-lbfgsb15_ids.txt \

QUADCOPTER_COLD_RESULTS := \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-tr-newton_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-panoc200_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-panoc50_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-ipopt-precompiled_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-strucpanoc50_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-gn_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-panoc20_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-lbfgsb15_ids.txt \

HANGING_CHAIN_RESULTS := \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-tr-newton_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-panoc200_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-panoc50_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-panoc20_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-ipopt_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-strucpanoc50_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-gn_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-lbfgsb15_ids.txt \

HANGING_CHAIN_COLD_RESULTS := \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-tr-newton_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-panoc200_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-panoc50_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-panoc20_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-ipopt_ids.txt \
    $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-strucpanoc50_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-gn_ids.txt \
    # $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-lbfgsb15_ids.txt \

.PHONY: all clean
all : $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-med-runtimes.pdf \
      $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-med-runtimes.pdf \
      $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-med-runtimes.pdf \
      $(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-med-runtimes.pdf \
      $(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-solution.pdf

# Quadcopter
$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-tr-newton_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		tr-newton fbetrust quadcopter ss2p $(DELTA) \
		dir.exact_newton=1 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-gn_ids.txt : run-ocp-mpc-exp.sh
	./$< $(OUTPUT) \
		gn panococp quadcopter $(DELTA) \
		solver.lbfgs_params.memory=50 solver.gn_interval=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-panoc200_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		panoc200 panoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=200 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-panoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		panoc50 panoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=50 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-panoc20_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		panoc20 panoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=20 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-strucpanoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		strucpanoc50 strucpanoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=50 dir.hessian_vec=0 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-lbfgsb15_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		lbfgsb15 lbfgsb quadcopter ss2p $(DELTA) \
		solver.memory=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-ipopt-precompiled_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		ipopt-precompiled ipopt quadcopter ss2 $(DELTA) \
		solver.warm_start_init_point=yes \
		ipopt.tol=$(TOL) ipopt.constr_viol_tol=$(TOL) warm=1 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-med-runtimes.pdf : perfplot.py $(QUADCOPTER_RESULTS)
	python3 $< $(PWD)/$(OUTPUT) mpc-Δ$(DELTA)-quadcopter $(DELTA) \
		"PANTR:mpc-Δ$(DELTA)-quadcopter-tr-newton" \
		"PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-panoc50" \
		"Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-strucpanoc50" \
		"Ipopt:mpc-Δ$(DELTA)-quadcopter-ipopt-precompiled" \
		# "PANOC GN:mpc-Δ$(DELTA)-quadcopter-gn" \
		# "PANOC$$^+$$ (200):mpc-Δ$(DELTA)-quadcopter-panoc200" \
		# "PANOC(200):mpc-Δ$(DELTA)-quadcopter-panoc200" \
		# "LBFGSB(15):mpc-Δ$(DELTA)-quadcopter-lbfgsb15"

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-solution.pdf : visualize_mpc_solution.py $(QUADCOPTER_RESULTS)
	python3 $< $(PWD)/$(OUTPUT) mpc-Δ$(DELTA)-quadcopter \
		mpc-Δ$(DELTA)-quadcopter-tr-newton

# Quadcopter (cold)
$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-tr-newton_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-tr-newton fbetrust quadcopter ss2p $(DELTA) \
		dir.exact_newton=1 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-gn_ids.txt : run-ocp-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-gn panococp quadcopter $(DELTA) \
		solver.lbfgs_params.memory=50 solver.gn_interval=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-panoc200_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-panoc200 panoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=200 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-panoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-panoc50 panoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=50 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-panoc20_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-panoc20 panoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=20 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-strucpanoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-strucpanoc50 strucpanoc quadcopter ss2p $(DELTA) \
		lbfgs.memory=50 dir.hessian_vec=0 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-lbfgsb15_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-lbfgsb15 lbfgsb quadcopter ss2p $(DELTA) \
		solver.memory=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e4 solver.max_iter=250 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-ipopt-precompiled_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-ipopt-precompiled ipopt quadcopter ss2 $(DELTA) \
		solver.warm_start_init_point=no \
		ipopt.tol=$(TOL) ipopt.constr_viol_tol=$(TOL) warm=0 num_dist=0
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-cold-med-runtimes.pdf : perfplot.py $(QUADCOPTER_COLD_RESULTS)
	python3 $< $(PWD)/$(OUTPUT) mpc-Δ$(DELTA)-quadcopter-cold $(DELTA) \
		"PANTR:mpc-Δ$(DELTA)-quadcopter-cold-tr-newton" \
		"PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-cold-panoc50" \
		"Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-cold-strucpanoc50" \
		"Ipopt:mpc-Δ$(DELTA)-quadcopter-cold-ipopt-precompiled" \
		# "PANOC GN:mpc-Δ$(DELTA)-quadcopter-cold-gn" \
		# "PANOC$$^+$$ (200):mpc-Δ$(DELTA)-quadcopter-cold-panoc200" \
		# "PANOC(200):mpc-Δ$(DELTA)-quadcopter-cold-panoc200" \
		# "LBFGSB(15):mpc-Δ$(DELTA)-quadcopter-cold-lbfgsb15"

$(OUTPUT)/mpc-Δ$(DELTA)-quadcopter-avg-runtimes-cold-warm.pdf : perfplot-cold-warm.py $(QUADCOPTER_RESULTS) $(QUADCOPTER_COLD_RESULTS)
	python3 $< $(PWD)/$(OUTPUT) mpc-Δ$(DELTA)-quadcopter $(DELTA) \
		"cold:PANTR:mpc-Δ$(DELTA)-quadcopter-cold-tr-newton" \
		"cold:PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-cold-panoc50" \
		"cold:Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-cold-strucpanoc50" \
		"cold:Ipopt:mpc-Δ$(DELTA)-quadcopter-cold-ipopt-precompiled" \
		"warm:PANTR:mpc-Δ$(DELTA)-quadcopter-tr-newton" \
		"warm:PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-panoc50" \
		"warm:Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-quadcopter-strucpanoc50" \
		"warm:Ipopt:mpc-Δ$(DELTA)-quadcopter-ipopt-precompiled"

# Hanging chain
$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-tr-newton_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		tr-newton fbetrust hanging_chain ss2p $(DELTA) \
		dir.exact_newton=1 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-gn_ids.txt : run-ocp-mpc-exp.sh
	./$< $(OUTPUT) \
		gn panococp hanging_chain $(DELTA) \
		solver.lbfgs_params.memory=50 solver.gn_interval=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-panoc200_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		panoc200 panoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=200 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-panoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		panoc50 panoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=50 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-panoc20_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		panoc20 panoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=20 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-strucpanoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		strucpanoc50 strucpanoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=50 dir.hessian_vec=0 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-lbfgsb15_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		lbfgsb15 lbfgsb hanging_chain ss2p $(DELTA) \
		solver.memory=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-ipopt_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		ipopt ipopt dl:hanging_chain ss2 $(DELTA) \
		solver.warm_start_init_point=yes \
		ipopt.tol=$(TOL) ipopt.constr_viol_tol=$(TOL) warm=1 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-med-runtimes.pdf : perfplot.py $(HANGING_CHAIN_RESULTS)
	python3 $< $(PWD)/$(OUTPUT) mpc-Δ$(DELTA)-hanging_chain $(DELTA) \
		"PANTR:mpc-Δ$(DELTA)-hanging_chain-tr-newton" \
		"PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-panoc50" \
		"Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-strucpanoc50" \
		"Ipopt (VM):mpc-Δ$(DELTA)-dl:hanging_chain-ipopt" \
		# "PANOC GN:mpc-Δ$(DELTA)-hanging_chain-gn" \
		# "LBFGSB(15):mpc-Δ$(DELTA)-hanging_chain-lbfgsb15"

# Hanging chain (cold)
$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-tr-newton_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-tr-newton fbetrust hanging_chain ss2p $(DELTA) \
		dir.exact_newton=1 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-gn_ids.txt : run-ocp-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-gn panococp hanging_chain $(DELTA) \
		solver.lbfgs_params.memory=50 solver.gn_interval=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-panoc200_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-panoc200 panoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=200 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-panoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-panoc50 panoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=50 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-panoc20_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-panoc20 panoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=20 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-strucpanoc50_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-strucpanoc50 strucpanoc hanging_chain ss2p $(DELTA) \
		lbfgs.memory=50 dir.hessian_vec=0 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-lbfgsb15_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-lbfgsb15 lbfgsb hanging_chain ss2p $(DELTA) \
		solver.memory=15 \
		alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 \
		alm.ε=$(TOL) alm.δ=$(TOL) alm.max_time=30s warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-ipopt_ids.txt : run-mpc-exp.sh
	./$< $(OUTPUT) \
		cold-ipopt ipopt dl:hanging_chain ss2 $(DELTA) \
		solver.warm_start_init_point=no \
		ipopt.tol=$(TOL) ipopt.constr_viol_tol=$(TOL) warm=0 \
		num_dist=5 u_dist=-1,1,1
	touch $@

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-cold-med-runtimes.pdf : perfplot.py $(HANGING_CHAIN_COLD_RESULTS)
	python3 $< $(PWD)/$(OUTPUT) mpc-Δ$(DELTA)-hanging_chain-cold $(DELTA) \
		"PANTR:mpc-Δ$(DELTA)-hanging_chain-cold-tr-newton" \
		"PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-cold-panoc50" \
		"Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-cold-strucpanoc50" \
		"Ipopt (VM):mpc-Δ$(DELTA)-dl:hanging_chain-cold-ipopt" \
		# "PANOC GN:mpc-Δ$(DELTA)-hanging_chain-cold-gn" \
		# "PANOC(20):mpc-Δ$(DELTA)-hanging_chain-cold-panoc20" \
		# "PANOC(200):mpc-Δ$(DELTA)-hanging_chain-cold-panoc200" \
		# "LBFGSB(15):mpc-Δ$(DELTA)-hanging_chain-cold-lbfgsb15"

$(OUTPUT)/mpc-Δ$(DELTA)-hanging_chain-avg-runtimes-cold-warm.pdf : perfplot-cold-warm.py $(HANGING_CHAIN_RESULTS) $(HANGING_CHAIN_COLD_RESULTS)
	python3 $< $(PWD)/$(OUTPUT) mpc-Δ$(DELTA)-hanging_chain $(DELTA) \
		"cold:PANTR:mpc-Δ$(DELTA)-hanging_chain-cold-tr-newton" \
		"cold:PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-cold-panoc50" \
		"cold:Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-cold-strucpanoc50" \
		"cold:Ipopt (VM):mpc-Δ$(DELTA)-dl:hanging_chain-cold-ipopt" \
		"warm:PANTR:mpc-Δ$(DELTA)-hanging_chain-tr-newton" \
		"warm:PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-panoc50" \
		"warm:Struc. PANOC$$^+$$ (50):mpc-Δ$(DELTA)-hanging_chain-strucpanoc50" \
		"warm:Ipopt (VM):mpc-Δ$(DELTA)-dl:hanging_chain-ipopt"
