using Random, LinearAlgebra, Plots, Printf

"""
    problem_data()
Returns the Q, q, a, and b matrix/vectors that defines the problem in Hand-In 1.
"""
function problem_data()
	mt = MersenneTwister(123)

	n = 20

	Qv = randn(mt,n,n)
	Q = Qv'*Qv
	q = randn(mt,n)

	a = -rand(mt,n)
	b = rand(mt,n)

	return Q,q,a,b
end

function primal_solver(step_size, fix_initial=1)
	Q,q,a,b = problem_data()

	L = norm(Q)
	gamma = step_size*2/L # gamma < 2/L according to instruction

	f_save = []


	if fix_initial == 1
		x_new = 1/2*(a+b)
	elseif fix_initial == 2
		θ = rand(20)
		x_new = a.*θ + b.*(1 .- θ)
	else
		x_new = b + a
	end
	for i in 1:20
		#@printf("x%3d: %3.3f \n", i, x_new[i])
	end
	x_old = a-b
	while (norm(x_old-x_new) > 10^-15) & (primal_target(x_new,Q,q,a,b) <= primal_target(x_old,Q,q,a,b))
		x_old = x_new
		f_save = [f_save; primal_target(x_old,Q,q,a,b)]
		x_new = prox_box(x_old - gamma*grad_quad(x_old,Q,q),a,b,gamma)
	end

	#println(@sprintf "Number of steps: %i" length(f_save))
	#println(@sprintf "ι(x):  %f" box(x_new,a,b))
	#println(-(x_new.==a) + (x_new.==b))
	if length(f_save) > 0
		#plot(f_save)
		#plot(log.(f_save .- minimum(f_save) .+ 1))
	end
	return length(f_save), x_new
end

function dual_solver(step_size, fix_initial=true)
	Q,q,a,b = problem_data()

	L = norm(inv(Q))
	gamma = step_size*2/L # gamma < 2/L according to instruction

	f_save = []

	if fix_initial
		s_new = zeros(20)
	else
		θ = rand(20)
		s_new = a.*θ + b.*(1 .- θ)
	end
	for i in 1:20
		#@printf("x%3d: %3.3f \n", i, x_new[i])
	end
	s_old = a-b
	while (norm(s_old-s_new) > 10^-15) & (dual_target(s_new,Q,q,a,b) <= dual_target(s_old,Q,q,a,b))
		s_old = s_new
		f_save = [f_save; dual_target(s_old,Q,q,a,b)]
		s_new = prox_boxconj(s_old - gamma*grad_quadconj(s_old,Q,q),a,b,gamma)
	end

	#println(@sprintf "Number of steps: %i" length(f_save))
	#println(@sprintf "ι(x):  %f" box(x_new,a,b))
	#println(-(x_new.==a) + (x_new.==b))
	if length(f_save) > 0
		#plot(f_save)
		#plot(log.(f_save .- minimum(f_save) .+ 1))
	end
	return length(f_save), s_new
end

function primal_results()
	x_ref = primal_solver(1, 3)[2]
	step_sizes = 0.1:0.1:1.6
	n_steps = []
	for ss in step_sizes
		ns, x  = primal_solver(ss)
		n_steps = [n_steps; ns]
		@printf("Step size: %1.1f \t Number of steps: %4d \t Distance from x_ref: %5.5f \n", ss, ns, norm(x-x_ref))
 	end
	display(plot(step_sizes, n_steps))
end

function dual_results()
	s_ref = dual_solver(1)[2]
	step_sizes = 0.1:0.1:1.6
	n_steps = []
	for ss in step_sizes
		ns, s  = dual_solver(ss)
		n_steps = [n_steps; ns]
		@printf("Step size: %1.1f \t Number of steps: %4d \t Distance from x_ref: %5.5f \n", ss, ns, norm(s-s_ref))
 	end
	plot(step_sizes, n_steps)
end

primal_results()
dual_results()
