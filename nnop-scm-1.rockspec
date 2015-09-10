package = "nnop"
version = "scm-1"

source = {
   url = "",
   branch = "master",
}

description = {
   summary = "NN Modules with operations only (no data).",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "totem",
   "nn",
}

build = {
   type = "builtin",
   modules = {
      ['nnop.init'] = 'init.lua',
      ['nnop.Linear'] = 'Linear.lua',
      ['nnop.Parameters'] = 'Parameters.lua',
      ['nnop.test'] = 'test.lua',
   },
}
