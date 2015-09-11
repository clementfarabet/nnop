-- deps:
require 'nn'
require 'nngraph'

-- create global cxnn table:
nnop = {}

-- modules:
require('./Module')
require('./Linear')

-- special node: parameters
require('./Parameters')

-- tests
nnop.test = function()
   require('./test')
end

return nnop
