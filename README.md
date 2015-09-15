# nnop

Parameter-free / operation-only Neural Network primitives
for torch/nn.

## Motivation, Goals

Sometimes, it's useful to treat parameters as regular states,
to either impose certain constraints on them, or simply
make weight sharing visible / straight-forward.

The original design of [nn](https://github.com/torch/nn) treats
trainable parameters as special variables. This package, `nnop`,
builds on `nn` and `nngraph`, but separates parameters from operations.

It introduces a new module, `nn.Parameters`, which provides trainable
parameters, but does not do any computation. Every other parameterized
node (`nn.Linear`, `nn.SpatialConvolution`, ...) needs to be wrapped in
`nnop` to decouple trainable parameters, and become pure operation nodes.

## TODO

* wrap remaining parametrized nodes (`nn.SpatialConvolution`, ...)
* simplify/unify auto-generated parameterNodes?

## Examples

### Weight sharing

In this example, 2 modules are connected to a same set of trainable
parameters. This is weight sharing.

```lua
-- Create parameters:
linearWeight = nnop.Parameters(10,100)()
linearBias = nnop.Parameters(10)()

-- Create multiple layers, all connected to these parameters:
input1 = nn.Identity()()
input2 = nn.Identity()()
linear1 = nnop.Linear()({input1, linearWeight, linearBias})
linear2 = nnop.Linear()({input2, linearWeight, linearBias})

-- Graph:
graph = nn.gModule({input1,input2}, {linear1,linear2})

-- Tests:
res = graph:forward({torch.randn(100), torch.randn(100)})
assert(type(res) == 'table' and #res == 2)

input = torch.randn(100)
res = graph:forward({input, input})
assert(res[1]:dist( res[2] ) == 0)
```

### Penalty on a set of parameters

In this example, we add an L1 penalty on a set of weight.

When parameters are provided to the nnop.Linear constructor,
parameter nodes are automatically created (and automatically
connected in the graph!). We use this in this example, this
way we don't have to create the parameter nodes, but are still
free to access them and add a penalty on them.

```lua
-- create base modules:
linear1 = nnop.Linear(10,100)
tanh1 = nn.Tanh()
linear2 = nnop.Linear(100,2)

-- bind them in a graph:
input = nn.Identity()()
layer1 = linear1(input)
layer2 = tanh1(layer1)
layer3 = linear2(layer2)

-- get weights and impose penalty:
weight1 = linear1.parameterNodes.weightNode
sparse1 = nn.L1Penalty(.001)(weight1)

-- build final model:
model = nn.gModule({input}, {layer3})

-- train the model:
for i = 1,10 do
   input = torch.rand(10)
   output = model:forward(input)
   gradOutput = torch.rand(2)
   gradInput = model:updateGradInput(input, gradOutput)
   model:accGradParameters(input, gradOutput)
end
```
