-- Pkg to test
local nnop = require 'nnop'
local nngraph = require 'nngraph'

local totem = require 'totem'
local tester = totem.Tester()

-- Tests:
local tests = {
   Parameters = function()
      -- Create parameter:
      local linearWeight = nnop.Parameters(10,100)
      local linearBias = nnop.Parameters(10)

      -- Forward:
      tester:assert(linearWeight:forward():dim(), 2, 'incorrect nb of dims')
   end,

   Linear = function()
      -- Create parameter:
      local linearWeight = nnop.Parameters(10,100)
      local linearBias = nnop.Parameters(10)

      -- Create Linear layer:
      local linear = nnop.Linear()

      -- Forward:
      local res = linear:forward({torch.randn(100), linearWeight:forward(), linearBias:forward()})

      -- Test:
      tester:eq(res:dim(), 1, 'incorrect nb of dims')
      tester:eq(res:size(1), 10, 'incorrect size')

      -- Backward:
      local grads = linear:backward({torch.randn(100), linearWeight:forward(), linearBias:forward()}, res)
      linearWeight:backward(nil, grads[2])
      linearBias:backward(nil, grads[3])

      -- Second example: let Linear generate its parameters
      local linear = nnop.Linear(100,10)
      local parameters = linear.parameterNodes
      local res = linear:forward({torch.randn(100), parameters.weight:forward(), parameters.bias:forward()})

      -- Test:
      tester:eq(res:dim(), 1, 'incorrect nb of dims')
      tester:eq(res:size(1), 10, 'incorrect size')
   end,

   nngraph = function()
      -- create base modules:
      local linear1 = nnop.Linear(10,100)
      local tanh1 = nn.Tanh()
      local linear2 = nnop.Linear(100,2)

      -- bind them in a graph:
      local input = nn.Identity()()
      local layer1 = linear1({input, linear1.parameterNodes.weight(), linear1.parameterNodes.bias()})
      local layer2 = tanh1(layer1)
      local output = linear2({layer2, linear2.parameterNodes.weight(), linear2.parameterNodes.bias()})

      -- build final model:
      local model = nn.gModule({input}, {output})

      nngraph.setDebug(true)

      local input = torch.rand(10)
      local gradOutput = torch.rand(2)
      local output = model:forward(input)
      local gradInput = model:updateGradInput(input, gradOutput)
      model:accGradParameters(input, gradOutput)

      tester:eq(output:dim(), 1 , 'incorrect nb of dims')
      tester:eq(output:size(1), 2, 'incorrect output size')
      tester:eq(gradInput:size(), input:size(), 'gradInput wrong size')
   end
}

tester:add(tests):run()
