-- deps:
require 'nn'
require 'nngraph'

-- create global cxnn table:
nnop = {}

-- modules:
require('./Module')
require('./Linear')

-- special graph nodes: input and parameters
require('./Input')
require('./Parameters')

-- tests
nnop.test = function()
   require('./test')
end

return nnop
