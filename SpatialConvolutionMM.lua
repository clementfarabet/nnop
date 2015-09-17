local SpatialConvolutionMM, parent = torch.class('nnop.SpatialConvolutionMM', 'nnop.Module')

function SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.gradInput = {
      torch.Tensor(),
      torch.Tensor(),
      torch.Tensor(),
   }

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()

   if nInputPlane and nOutputPlane and kW and kH then
      self.parameterNodes = {
         weight = nnop.Parameters(nOutputPlane, nInputPlane*kH*kW),
         bias = nnop.Parameters(nOutputPlane),
      }
   end
end

function SpatialConvolutionMM:updateOutput(inputs)
   local input = inputs[1]
   local weight = inputs[2]
   local bias = inputs[3]
   nn.SpatialConvolutionMM.updateOutput({
      weight=weight, bias=bias,
      nInputPlane = self.nInputPlane, nOutputPlane = self.nOutputPlane,
      kW = self.kW, kH = self.kH,
      dW = self.dW, dH = self.dH,
      padW = self.padW, padH = self.padH,
      finput = self.finput, fgradInput = self.fgradInput,
      output = self.output
   }, input)
   return self.output
end

function SpatialConvolutionMM:updateGradInput(inputs, gradOutput)
   local input = inputs[1]
   local weight = inputs[2]
   local bias = inputs[3]
   assert(weight:dim() == 2, 'weight must have 2 dims')
   assert(weight:size(1) == bias:size(1), 'weight:size(1) == bias:size(1)')
   nn.SpatialConvolutionMM.updateGradInput({
      weight=weight, bias=bias,
      nInputPlane = self.nInputPlane, nOutputPlane = self.nOutputPlane,
      kW = self.kW, kH = self.kH,
      dW = self.dW, dH = self.dH,
      padW = self.padW, padH = self.padH,
      finput = self.finput, fgradInput = self.fgradInput,
      output = self.output,
      gradInput = self.gradInput[1]
   }, input, gradOutput)
   self.gradInput[2]:resizeAs(weight):zero()
   self.gradInput[3]:resizeAs(bias):zero()
   nn.SpatialConvolutionMM.accGradParameters({
      weight=weight, bias=bias,
      nInputPlane = self.nInputPlane, nOutputPlane = self.nOutputPlane,
      kW = self.kW, kH = self.kH,
      dW = self.dW, dH = self.dH,
      padW = self.padW, padH = self.padH,
      finput = self.finput, fgradInput = self.fgradInput,
      gradWeight=self.gradInput[2], gradBias=self.gradInput[3],
      output=self.output,
   }, input, gradOutput)
   return self.gradInput
end
