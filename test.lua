-- Pkg to test
local nnop = require 'nnop'

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
   end,
}

tester:add(tests):run()
