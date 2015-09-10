-- deps:
require 'nn'

-- create global cxnn table:
nnop = {}

-- modules:
require('./Linear')

-- special node: parameters
require('./Parameters')

-- tests
nnop.test = function()
   require('./test')
end

return nnop
