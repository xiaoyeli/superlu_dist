# Global symbol without underscore.
set(FortranCInterface_GLOBAL_SYMBOL  "mysub_")
set(FortranCInterface_GLOBAL_PREFIX  "")
set(FortranCInterface_GLOBAL_SUFFIX  "_")
set(FortranCInterface_GLOBAL_CASE    "LOWER")
set(FortranCInterface_GLOBAL_MACRO   "(name,NAME) name##_")

# Global symbol with underscore.
set(FortranCInterface_GLOBAL__SYMBOL "my_sub_")
set(FortranCInterface_GLOBAL__PREFIX "")
set(FortranCInterface_GLOBAL__SUFFIX "_")
set(FortranCInterface_GLOBAL__CASE   "LOWER")
set(FortranCInterface_GLOBAL__MACRO  "(name,NAME) name##_")

# Module symbol without underscore.
set(FortranCInterface_MODULE_SYMBOL  "mymodule_mysub_")
set(FortranCInterface_MODULE_PREFIX  "")
set(FortranCInterface_MODULE_MIDDLE  "_")
set(FortranCInterface_MODULE_SUFFIX  "_")
set(FortranCInterface_MODULE_CASE    "LOWER")
set(FortranCInterface_MODULE_MACRO   "(mod_name,name, mod_NAME,NAME) mod_name##_##name##_")

# Module symbol with underscore.
set(FortranCInterface_MODULE__SYMBOL "my_module_my_sub_")
set(FortranCInterface_MODULE__PREFIX "")
set(FortranCInterface_MODULE__MIDDLE "_")
set(FortranCInterface_MODULE__SUFFIX "_")
set(FortranCInterface_MODULE__CASE   "LOWER")
set(FortranCInterface_MODULE__MACRO  "(mod_name,name, mod_NAME,NAME) mod_name##_##name##_")

# Summarize what was found.
set(FortranCInterface_GLOBAL_FOUND 1)
set(FortranCInterface_MODULE_FOUND 1)

