local Module, parent = torch.class('nnop.Module', 'nn.Module')

function Module:__call__(...)
   local args = {...}
   if type(args[1]) == 'table' and #args[1] == 0 and self.parameterNodes then
      -- auto-connect parameters:
      local nodes = {args[1]}
      if self.parameterNodes.weight then
         self.parameterNodes.weightNode = self.parameterNodes.weight()
         table.insert(nodes, self.parameterNodes.weightNode)
      end
      if self.parameterNodes.bias then
         self.parameterNodes.biasNode = self.parameterNodes.bias()
         table.insert(nodes, self.parameterNodes.biasNode)
      end
      return parent.__call__(self,nodes)
   else
      return parent.__call__(self,...)
   end
end
