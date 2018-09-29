using StaticArrays
using Distributions
using Setfield

abstract type Particle end
abstract type Process end

struct Photon <: Particle
    E::Float64
    position::SVector{3,Float64}
    direction::SVector{3,Float64}
end

struct Electron <: Particle
    E::Float64
    position::SVector{3,Float64}
    direction::SVector{3,Float64}
end

struct Positron <: Particle
    E::Float64
    position::SVector{3,Float64}
    direction::SVector{3,Float64}
end

struct PhotoEffect <: Process end
struct ComptonEffect <: Process end

struct State
    photons::Vector{Photon}
    electrons::Vector{Electron}
end

function stack(::Type{Photon}, s::State)
    s.photons
end
function stack(::Type{Electron}, s::State)
    s.electrons
end

function processes(::Type{Photon}, s::State)
    PhotoEffect(), ComptonEffect()
end

Base.push!(s::State, p) = push!(stack(typeof(p), s), p)
Base.pop!(s::State, ::Type{P}) where {P} = pop!(stack(P, s))

function simulate_event!(state::State, p_init)
    push!(state, p_init)
    while true
        if !isempty(stack(Photon, state))
            cutoff_or_step!(state, pop!(stack(Photon, state)))
        elseif !isempty(stack(Electron, state))
            cutoff_or_step!(state, pop!(stack(Electron, state)))
        else
            break
        end
    end
end

function mean_free_path(state, p::Photon, proc::PhotoEffect)
    1/(p.E)^3
end

function mean_free_path(state, p::Photon, proc::ComptonEffect)
    p.E < 0.5 ? 1000. : 1.
end

function roll_stepsize(state, p::Particle, proc::Process)
    lambda = mean_free_path(state,p,proc)
    rand(Exponential(lambda))
end

function roll_iproc_stepsize(state, p::Particle)
    procs = processes(typeof(p), state)
    stepsizes = map(procs) do proc
        roll_stepsize(state, p, proc)
    end
    iproc = argmax(stepsizes)
    stepsize = stepsizes[iproc]
    # runtime dispatch
    iproc, stepsize
end

function dispatch_process(f, procs, i)
    if i == 1
        f(procs[1])
    elseif i == 2
        f(procs[2])
    elseif i == 3
        f(procs[3])
    else
        error()
    end
    nothing
end

function run_process!(state,p::Photon,proc::ComptonEffect, stepsize)
    E2 = rand() * p.E
    E1 = p.E - E2
    pos = p.position + stepsize * p.direction
    dir1 = @SVector randn(3)
    dir2 = @SVector randn(3)
    push!(state, Electron(E1, pos, dir1))
    push!(state, Photon(E2, pos, dir2))
end
function run_process!(state, p::Photon,proc::PhotoEffect, stepsize)
    E2 = rand() * p.E
    E1 = p.E - E2
    pos = p.position + stepsize * p.direction
    dir1 = @SVector randn(3)
    dir2 = @SVector randn(3)
    push!(state, Electron(E1, pos, dir1))
    push!(state, Photon(E2, pos, dir2))
end

function needscutoff(state, p::Photon)
    p.E < 0.1
end
function needscutoff(state, p::Particle)
    true
end
function edep!(state, p, E)
    println("edep!($state, $p)")
end

function cutoff_or_step!(state, p)
    if needscutoff(state,p)
        edep!(state, p, p.E)
    else
        step!(state,p)
    end
end

function step!(state, p::Photon)
    i, stepsize = roll_iproc_stepsize(state, p)
    procs = processes(Photon, state)
    dispatch_process(procs, i) do proc
        run_process!(state, p, proc, stepsize)
    end
end
