-- Pkg to test
local nnop = require 'nnop'
local nngraph = require 'nngraph'

-- Tester:
local totem = require 'totem'
local tester = totem.Tester()

-- For Jacobian tests:
local nn = require 'nn'
local Jacobian = nn.Jacobian

-- Precision for Jacobian tests:
local precision = 1e-5

-- Tests:
local tests = {
   Parameters = function()
      -- Create parameter:
      local linearWeight = nnop.Parameters(10,100)
      local linearBias = nnop.Parameters(10)

      -- Forward:
      tester:asserteq(linearWeight:forward():dim(), 2, 'incorrect nb of dims')
   end,

   LinearBasic = function()
      -- Create parameter:
      local linearWeight = nnop.Parameters(10,100)
      local linearBias = nnop.Parameters(10)

      -- Create Linear layer:
      local linear = nnop.Linear()

      -- Forward:
      local res = linear:forward({torch.randn(100), linearWeight:forward(), linearBias:forward()})

      -- Test:
      tester:asserteq(res:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res:size(1), 10, 'incorrect size')

      -- Backward:
      local grads = linear:backward({torch.randn(100), linearWeight:forward(), linearBias:forward()}, res)
      linearWeight:backward(nil, grads[2])
      linearBias:backward(nil, grads[3])

      -- Second example: let Linear generate its parameters
      local linear = nnop.Linear(100,10)
      local parameters = linear.parameterNodes
      local res = linear:forward({torch.randn(100), parameters.weight:forward(), parameters.bias:forward()})

      -- Test:
      tester:asserteq(res:dim(), 1, 'incorrect nb of dims')
      tester:asserteq(res:size(1), 10, 'incorrect size')
   end,

   LinearGraph = function()
      -- create base modules:
      local linear1 = nnop.Linear(10,100)
      local tanh1 = nn.Tanh()
      local linear2 = nnop.Linear(100,2)

      -- bind them in a graph:
      local input = nnop.Input()()
      local layer1 = linear1({input, linear1.parameterNodes.weight(), linear1.parameterNodes.bias()})
      local layer2 = tanh1(layer1)
      local output = linear2({layer2, linear2.parameterNodes.weight(), linear2.parameterNodes.bias()})

      -- build final model:
      local model = nn.gModule({input}, {output})

      -- geometry tests:
      local input = torch.rand(10)
      local gradOutput = torch.rand(2)
      local output = model:forward(input)
      local gradInput = model:updateGradInput(input, gradOutput)
      model:accGradParameters(input, gradOutput)

      tester:asserteq(output:dim(), 1 , 'incorrect nb of dims')
      tester:asserteq(output:size(1), 2, 'incorrect output size')
      tester:asserteq(gradInput:size(1), input:size(1), 'gradInput wrong size')

      -- look at parameters:
      local parameters,gradParameters = model:parameters()
      tester:asserteq(#parameters, 4, 'incorrect nb of parameters')
      tester:asserteq(#parameters, #gradParameters , 'incorrect nb of gradients')

      -- test zeroing:
      for i,gradParams in ipairs(gradParameters) do
         gradParams:fill(1)
      end
      model:zeroGradParameters()
      for i,gradParams in ipairs(gradParameters) do
         tester:asserteq(gradParams:sum(), 0, 'error on zeroGradParameters()')
      end
   end,

   LinearJacobian = function()
      -- create base modules:
      local linear1 = nnop.Linear(10,100)
      local tanh1 = nn.Tanh()
      local linear2 = nnop.Linear(100,2)

      -- bind them in a graph:
      local input = nnop.Input()()
      local layer1 = linear1({input, linear1.parameterNodes.weight(), linear1.parameterNodes.bias()})
      local layer2 = tanh1(layer1)
      local output = linear2({layer2, linear2.parameterNodes.weight(), linear2.parameterNodes.bias()})

      -- build final model:
      local model = nn.gModule({input}, {output})

      -- Test backprop:
      local input = torch.Tensor(10):zero()
      local err = Jacobian.testJacobian(model, input)
      tester:assertlt(err, precision, 'error on gradInput')

      -- Test internal weights:
      local parameters,gradParameters = model:parameters()
      for i = 1,#parameters do
         local err = Jacobian.testJacobianParameters(model, input, parameters[1], gradParameters[1])
         tester:assertlt(err, precision, 'error on gradParameters['..i..']')
      end
   end,

   LinearGraphAutoParams = function()
      -- bind them in a graph:
      local input = nnop.Input()()
      local layer1 = nnop.Linear(10,100)(input)
      local layer2 = nn.Tanh()(layer1)
      local layer3 = nnop.Linear(100,2)(layer2)

      -- build final model:
      local model = nn.gModule({input}, {layer3})

      -- geometry tests:
      local input = torch.rand(10)
      local gradOutput = torch.rand(2)
      local output = model:forward(input)
      local gradInput = model:updateGradInput(input, gradOutput)
      model:accGradParameters(input, gradOutput)

      tester:asserteq(output:dim(), 1 , 'incorrect nb of dims')
      tester:asserteq(output:size(1), 2, 'incorrect output size')
      tester:asserteq(gradInput:size(1), input:size(1), 'gradInput wrong size')

      -- play with parameters:
      layer1.data.module.parameterNodes.weight:uniform(-1,1)
      layer1.data.module.parameterNodes.bias:uniform(-1,1)
      layer3.data.module.parameterNodes.weight:uniform(-2,2)
      layer3.data.module.parameterNodes.bias:uniform(-2,2)
      tester:assertlt(layer1.data.module.parameterNodes.weight.weight:max(), 1.01, 'incorrect initialization')
      tester:assertgt(layer1.data.module.parameterNodes.bias.weight:min(), -1.01, 'incorrect initialization')
      tester:assertlt(layer3.data.module.parameterNodes.weight.weight:max(), 2.01, 'incorrect initialization')
      tester:assertgt(layer3.data.module.parameterNodes.bias.weight:min(), -2.01, 'incorrect initialization')
   end,

   LinearGraphWeightLoss = function()
      -- create base modules:
      local linear1 = nnop.Linear(10,100)
      local tanh1 = nn.Tanh()
      local linear2 = nnop.Linear(100,2)

      -- bind them in a graph:
      local input = nnop.Input()()
      local layer1 = linear1(input)
      local layer2 = tanh1(layer1)
      local layer3 = linear2(layer2)

      -- get weights:
      local weight1 = linear1.parameterNodes.weightNode
      local sparse1 = nn.L1Penalty(.001)(weight1)

      -- build final model:
      local model = nn.gModule({input}, {layer3})

      -- geometry tests:
      local input = torch.rand(10)
      local output = model:forward(input)
      local gradOutput = torch.rand(2)
      local gradInput = model:updateGradInput(input, gradOutput)
      model:accGradParameters(input, gradOutput)
   end,

   SpatialConvolutionMMJacobian = function()
      -- create base modules:
      local conv = nnop.SpatialConvolutionMM(4,16,5,5)

      -- bind them in a graph:
      local input = nnop.Input()()
      local output = conv({input, conv.parameterNodes.weight(), conv.parameterNodes.bias()})

      -- build final model:
      local model = nn.gModule({input}, {output})

      -- Test backprop:
      local input = torch.Tensor(4,10,10):zero()
      local err = Jacobian.testJacobian(model, input)
      tester:assertlt(err, precision, 'error on gradInput')

      -- Test internal weights:
      local parameters,gradParameters = model:parameters()
      for i = 1,#parameters do
         local err = Jacobian.testJacobianParameters(model, input, parameters[1], gradParameters[1])
         tester:assertlt(err, precision, 'error on gradParameters['..i..']')
      end
   end,
}

tester:add(tests):run()
