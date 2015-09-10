local Parameters, parent = torch.class('nnop.Parameters', 'nn.Module')

function Parameters:__init(...)
   parent.__init(self)
   self.weight = torch.Tensor(...)
   self.gradWeight = torch.Tensor(...)
end

function Parameters:updateOutput()
   self.output = self.weight
   return self.output
end

function Parameters:updateGradInput(_, gradOutput)
   self.gradInput = gradOutput
end

function Parameters:accGradParameters(_, gradOutput, scale)
   self.gradWeight:add(scale, self.gradInput)
end
