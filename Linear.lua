local Linear, parent = torch.class('nnop.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize)
   parent.__init(self)
   self.gradInput = {
      torch.Tensor(),
      torch.Tensor(),
      torch.Tensor(),
   }
   if inputSize and outputSize then
      self.parameters = {
         nnop.Parameters(outputSize, inputSize),
         nnop.Parameters(outputSize)
      }
   end
end

function Linear:updateOutput(inputs)
   local input = inputs[1]
   local weight = inputs[2]
   local bias = inputs[3]
   assert(weight:dim() == 2, 'weight must have 2 dims')
   assert(weight:size(1) == bias:size(1), 'weight:size(1) == bias:size(1)')
   nn.Linear.updateOutput({
      weight=weight, bias=bias,
      output=self.output
   }, input)
   return self.output
end

function Linear:updateGradInput(inputs, gradOutput)
   local input = inputs[1]
   local weight = inputs[2]
   local bias = inputs[3]
   assert(weight:dim() == 2, 'weight must have 2 dims')
   assert(weight:size(1) == bias:size(1), 'weight:size(1) == bias:size(1)')
   nn.Linear.updateGradInput({
      weight=weight, bias=bias,
      output=self.output,
      gradInput=self.gradInput[1]
   }, input, gradOutput)
   self.gradInput[2]:resizeAs(weight):zero()
   self.gradInput[3]:resizeAs(bias):zero()
   nn.Linear.accGradParameters({
      weight=weight, bias=bias,
      gradWeight=self.gradInput[2], gradBias=self.gradInput[3],
      output=self.output,
   }, input, gradOutput)
   return self.gradInput
end
