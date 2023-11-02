# EventOnsetDecoding.jl
Predict events at specified latencies from neural population responses

![ci](https://github.com/grero/EventOnsetDecoding.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/grero/EventOnsetDecoding.jl/branch/main/graph/badge.svg?token=tgY2I9vIht)](https://codecov.io/gh/grero/EventOnsetDecoding.jl)

## Installation

First, download the latest stable julia release from [https://julialang.org/downloads](https://julialang.org/downloads/).

Then, start julia, enter the package manager by pressing `]` and add the NeuralCoding registry

```julia
(@v1.9) pkg> registry add https://github.com/grero/NeuralCodingRegistry.jl
```
Once the registry has been succesfully added, you should be able to install this pacakge by doing

```julia
(@v1.9) pkg> add EventOnsetDecoding
```

This will install the package under the default environment (similar to conda environment). If you'd rather create a
new enviroment, e.g. MyCoolAnalysis, you can first create a new directory for your new environment, change to that directory and activate it, with tese commands, making use of julia's shell mode (press `;`)

```julia
shell> mkdir MyCoolAnalysis
shell> cd MyCoolAnalysis
(@v1.9) pkg> activate .
```
To move from shell mode back to package manager mode in he last line, simply press `]` again.

Now, whenver you `add` stuff in package mananger mode, it will be added to the environment contained in MyCoolAnalysis. To run julia directory from this environment, you can start julia like you did previously, but add the option --project=., i.e.

```bash
julia --project=.
```


## Usage

A basic Jupyter notebook is include in notebooks/basic_tutorial.ipynb. This works just like a python notebook, except that it runs julia code. To enable julia for jupyter notebooks, it simplest way is to install the package IJulia in the global julia environment, like this

```julia
(@v1.9) pkg> add IJulia 
```
The global environment can be accessed by starting julia without any options like we did before, or by using the `activate` command with no arguments, i.e.

```julia
(MyCoolAnalysis) pkg> activate
(@v1.9) pkg>

```
After installing IJulia, you can also start up a notebook server directly from julia

```julia
julia> using IJulia
julia> notebook()
```

The `using` statement in the first line is sort of the equivalent of python's `import` statement. It makes the code contained in the `IJulia` package available in the current session.

