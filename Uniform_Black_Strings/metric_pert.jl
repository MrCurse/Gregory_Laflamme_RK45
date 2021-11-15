"""
Numerical tests on Gregory-Laflamme instability of metric components of uniform
black strings using Julia's DifferentialEquations package
"""
#using Gadfly
using PyPlot
pygui(true)
using DifferentialEquations

function V(r, D, rp=2.0)
    return @. 1 - (rp/r)^(D-3)
end

function P(r, μ, Ω, D, rp=2.0)
    num = @. μ*μ*((D-2)-2*(rp/r)^(D-3)+(4-D)*(rp/r)^(2*D-6))/(r*V(r,D))+Ω*Ω*((D-2) + (2*D-7)*(rp/r)^(D-3))/(r*V(r,D))-( 3*(D-3)^2 *(rp/r)^(2*D-6)
                *((D-2)-(rp/r)^(D-3)) )/(4*V(r,D)*r^3)
    denom = @. -Ω^2 - μ^2 *V(r,D) + ((D-3)^2 *(rp/r)^(2*D-6))/(4*r^2)

    return @. num/denom
end

function Q(r, μ, Ω, D, rp=2.0)
    num = @.( (μ^2 +Ω^2 /V(r,D))^2 +(Ω^2 /(4*r*r*V(r,D)^2))*(4*D-8-(8*D-16)*(rp/r)^(D-3) -(53-34*D+5*D^2)*(rp/r)^(2*D-6))+(μ^2 /(4*r*r*V(r,D)))
                *(4*D-8-4*(3*D-7)*(rp/r)^(D-3)+(D^2 +2*D-11)*(rp/r)^(2*D-6))+((D-3)^2 /(4*r^4 *V(r,D)^2))*(rp/r)^(2*D-6) *((D-2)*(2*D-5)
                         -(D-1)*(D-2)*(rp/r)^(D-3)+(rp/r)^(2*D-6)) )
    denom = @. -Ω^2 - μ^2 *V(r,D) + ((D-3)^2 *(rp/r)^(2*D-6))/(4*r^2)

    return @. -num/denom
end

function metric_pert(du, u, p, r)
    μ, Ω, D = p
    H = u[1]
    G = u[2]
    du[1] = G
    du[2] = P(r, μ, Ω, D)*G + Q(r, μ, Ω, D)*H
end

function compute_ratio(μ, D)

    rp::Float64 = 2.0
    r1::Float64 = 200.0
    r2::Float64 = 2.00002
    Ω = LinRange(0, 0.1, 500)
    ℛ = zeros(length(Ω))

    for i in 1:1:length(Ω)

        p = [μ, Ω[i], D]

        Htr0 = exp(-sqrt(μ^2 + Ω[i]^2)*r1)
        Gtr0 = -sqrt(μ^2 + Ω[i]^2)*Htr0

        u0 = [Htr0, Gtr0]
        tspan = [r1, r2]

        prob = ODEProblem(metric_pert, u0, tspan, p)
        """
        We use an algorithm switching method between a 9-th order Vern9() and a
        5-th order RadauIIA5() methods. Initially the solutions is computed using
        Vern9(), which is an explicit method. The switch to an implicit RadauIIA5()
        occurs when stiffness is detected in the evaluation.

        We also set a maximum stepsize (dtmax) to prevent the solutions overshooting
        for certain values of μ[i].
        """
        sol = solve(prob, AutoVern9(RadauIIA5()), reltol=10^-10, abstol=10^-10, dtmax=10, maxiters=10^7)


        r = sol.t
        Htr = sol[1,:]
        Gtr = sol[2,:]

        #can be used as an alternative expression for ℛ
        ℛ[i] = (r2^(D-3) - rp^(D-3))^(2*rp*Ω[i]/(D-3)) *( (Htr[end]*(rp*Ω[i] - D + 3) - r2^(4-D) *(r2^(D-3) - rp^(D-3))*Gtr[end])/(Htr[end]*(rp*Ω[i] + D - 3)
                       + r2^(4-D) *(r2^(D-3) - rp^(D-3))*Gtr[end]) )
        #ℛ[i] = ( (Htr[end]*(rp*Ω[i] - 1) - (r2 - rp)*Gtr[end])/(Htr[end]*(rp*Ω[i] + 1)
        #                    + (r2 - rp)*Gtr[end]) )*(r2 - rp)^((2*rp*Ω[i])/(D-3))

    end

    u_index = findall(ℛ.>0)[1]
    Ω_u = Ω[u_index]

    return Ω_u
end

μ = collect(0.05:0.05:0.45)
μ = pushfirst!(μ, 0.01)
#D = [4,5,6,7,8]
D = [4]
Ω = zeros(length(μ), length(D))

for j in 1:1:length(D)
    for i in 1:1:length(μ)
        """
        The try-catch block handles the BoundsError when the condition ℛ = 0
        cannot be realized for a particular value of μ[i]
        """
        try
            Ω[i,j] = compute_ratio(μ[i], D[j])
        catch
            println("exception occured at μ = " * string(μ[i]))
        end
    end
end

for i in 1:1:length(D)
    scatter(μ, Ω[:,i])
end
