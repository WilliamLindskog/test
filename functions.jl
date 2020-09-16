"""
    quad(x,Q,q)
Compute the quadratic
	1/2 x'Qx + q'x
"""
function quad(x,Q,q)
	return 1/2 * x'*Q*x + q'*x
end



"""
    guadconj(s,Q,q)
Compute the convex conjugate of the quadratic
	1/2 s'Qx + s'x
"""
function quadconj(s,Q,q)
	return (1/2) * (s-q)'*inv(Q)*(s-q)
end



"""
    box(x,a,b)
Compute the indicator function of for the box contraint
	a <= x <= b
where the inequalites are applied element-wise.
"""
function box(x,a,b)
	return all(a .<= x .<= b) ? 0.0 : Inf
end



"""
    boxconj(s,a,b)
Compute the convex conjugate of the indicator function for the box constraint
	a <= s <= b
where the inequalites are applied element-wise.
"""
function boxconj(s,a,b)
    return s'*(a.*(s.<=0) + b.*(s.>0))
end



"""
    grad_quad(x,Q,q)
Compute the gradient of the quadratic
	1/2 x'Qx + q'x
"""
function grad_quad(x,Q,q)
	return Q*x + q
end



"""
    grad_quadconj(s,Q,q)
Compute the gradient of the convex conjugate of the quadratic
	1/2 x'Qx + q'x
"""
function grad_quadconj(s,Q,q)
	return inv(Q)*(s - q)
end



"""
    prox_box(x,a,b)
Compute the proximal operator of the indicator function for the box constraint
	a <= x <= b
where the inequalites are applied element-wise.
"""
a = [0; 0]
b = [2; 2]
x = [1; 3]

function prox_box(x,a,b,gamma)
	return a.*(x.<a) + x.*(a.<= x.<=b) + b.*(b.<x)
end

prox_box(x,a,b,1)


"""
    prox_boxconj(y,a,b)
Compute the proximal operator of the convex conjugate of the indicator function
for the box contraint
	a <= x <= b
where the inequalites are applied element-wise.
"""
function prox_boxconj(z,a,b,gamma)
	return (z .- gamma*sum(a)).*(z.<gamma*sum(a)) + (z .- gamma*sum(b)).*(z.>gamma*sum(b))
end


"""
    dual2primal(y,Q,q,a,b)
Computes the solution to the primal problem for Hand-In 1 given a solution y to
the dual problem.
"""
function dual2primal(y,Q,q,a,b)
	return inv(Q)*(y - q)
end

function primal_target(x,Q,q,a,b)
	return quad(x,Q,q) + box(x,a,b)
end

function dual_target(s,Q,q,a,b)
	return quadconj(s,Q,q) + boxconj(s,a,b)
end
