local Input, parent = torch.class('nnop.Input', 'nn.Identity')

-- for now Input == Identity
-- this is more explicity to define inputs to a graph; and
-- a placeholder so we can extend the behavior of input nodes.
