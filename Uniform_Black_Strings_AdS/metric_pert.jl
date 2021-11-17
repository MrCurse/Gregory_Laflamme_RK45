using PyPlot
pygui(true)
using DifferentialEquations

function V(r, D, rp=2.0, 𝓁=100.0)
    return @. 1 - r^(4-D) *(rp^(4-D) + rp^(D-2) /𝓁^2) + (r/𝓁)^2
end

function P(r, ,μ, Ω, D, rp=2.0, 𝓁=100.0)
    num = @. ( (D-2)*r^2 + (4-D)*𝓁^2 )/( 3*(D-4)*(2*D-9)*𝓁^2 + r^2 *(-2+D*(6*D-11)+8*μ^2 *𝓁^2) )
            - 4*(2*D-9)*r^2 *Ω^2 *𝓁^4 - ( (3*D-6)*r^2 + (3*D-12)*𝓁^2 )/(V(r,D)*𝓁^2)
                *( ( (D-2)*r^2 + (D-4)*𝓁^2 )^2 - 4*r^2 *Ω^2 *𝓁^4 ) - V(r,D)*𝓁^2
                    *( (3*D-18)*(4-D)^2 *𝓁^2 +r^2 *(8+24*D-20*D^2 +3*D^3 +(4*D-20)*𝓁^2 *μ^2)
                        +3*(D-4)^2 *𝓁^2 *V(r,D) )

    denom = @. r*(( (D-2)*r^2 + (D-4)*𝓁^2 )^2 - 4*r^2 *Ω^2 *𝓁^4 - 2*𝓁^2 *( (D-4)^2 *𝓁^2
                +r^2 *(D^2 -4 +2*𝓁^2 *μ^2) )*V(r,D) + 𝓁^4 *V(r,D)^2 *(D-4)^2)

    return @. num/denom
end

function Q(r, ,μ, Ω, D, rp=2.0, 𝓁=100.0)
    num = @. -( (D-2)*r^2 + (D-4)*𝓁^2 )^4 + 5*r^2 *𝓁^4 *( (D-2)*r^2 + (D-4)*𝓁^2 )^2 *Ω^2
            - 4*r^4 *𝓁^8 *Ω^4 - 𝓁^2 *V(r,D)*( ( (D-2)*r^2 + (D-4)*𝓁^2 )^2 *( (7-D)
            *((D-2)*r^2 + (D-4)*𝓁^2) + r^2 *𝓁^2 *μ^2 ) + 2*r^2 *𝓁^4 *( 5*(D-4)^2 *𝓁^2
            +r^2 *(28-24*D+5*D^2 +4*𝓁^2 *μ^2) )*Ω^2 + V(r,D)*( -(D-4)^2 *(D^2 -2*D -9)*𝓁^6
            +r^4 *𝓁^2 *( -(D-2)^2 *(-13+2*D +D^2)-(2*D-16)*(D-2)*𝓁^2 *μ^2 + 4*𝓁^2 *μ^4 )
            +r^2 *𝓁^4 *( (8-2*D)*(D-2)*(-11+D^2 +𝓁^2 *μ^2)+(-92+44*D-5*D^2)*𝓁^2 *Ω^2 )
            +𝓁^4 *V(r,D)*( (D-4)^2 *(2-5*D+D^2)*𝓁^2 +r^2 *(56-56*D+24*D^2 -7*D^3 +D^4
            -12*𝓁^2 *μ^2 +D^2 *𝓁^2 *μ^2) +(D-4)^2 *𝓁^2 *V(r,D) ) ) )

    denom = @. r^2 *V(r,D)^2 *𝓁^4 *(( (D-2)*r^2 + (D-4)*𝓁^2 )^2 - 4*r^2 *Ω^2 *𝓁^4
            - 2*𝓁^2 *( (D-4)^2 *𝓁^2+r^2 *(D^2 -4 +2*𝓁^2 *μ^2) )*V(r,D) + 𝓁^4 *V(r,D)^2 *(D-4)^2)

    return @. num/denom
end

function metric_pert(du, u, p, r)
    μ, Ω, D = p
    H = u[1]
    G = u[2]
    du[1] = G
    du[2] = P(r, μ, Ω, D)*G + Q(r, μ, Ω, D)*H
end

function compute_ratio(μ, D)
    rp = 2.0
    r1 = 200.0
    r2 = 2.0001
    𝓁 = 100.0
    ε = ((D-2)*rp^2)/((D-4)*𝓁^2)
    Ω = LinRange(0.0, 0.2, 1000)
    ℛ = zeros(length(Ω))

    for i in 1:1:length(Ω)
        if μ > Ω

            p = [μ, Ω[i], D]

            Htr0 = exp(-r1*μ^2 /sqrt(\mu^2 - Ω^2))
            Gtr0 = -(μ^2 /sqrt(\mu^2 - Ω^2))*Htr0

            u0 = [Htr0, Gtr0]
            tspan = [r1, r2]

            prob = ODEProblem(metric_pert, u0, tspan, p)
            sol = solve(prob, AutoVern9(RadauIIA5()), reltol=10^-10, abstol=10^-10
                            dtmax=10, maxiters=10^7)

            r = sol.t
            Htr = sol[1,:]
            Gtr = sol[2,:]

            ℛ[i] = ( Htr[end]*(rp*Ω/(D-4)*sqrt(1-ε)-1) - (r2-rp2)*Gtr[end] )
                    /( Htr[end]*(rp*Ω/(D-4)*sqrt(1-ε)+1) + (r2-rp2)*Gtr[end] )
                        * (r2 - rp)^((2*rp*Ω/(D-4))*sqrt(1-ε))
            else
                println("Ω > μ...evaluation is invalid")

        end
    end
end
