local Module, parent = torch.class('nnop.Module', 'nn.Module')

function Module:__call__(...)
   local args = {...}
   if type(args[1]) == 'table' and #args[1] == 0 and self.parameterNodes then
      -- auto-connect parameters:
      local nodes = {args[1]}
      if self.parameterNodes.weight then
         table.insert(nodes, self.parameterNodes.weight())
      end
      if self.parameterNodes.bias then
         table.insert(nodes, self.parameterNodes.bias())
      end
      return parent.__call__(self,nodes)
   else
      return parent.__call__(self,...)
   end
end
