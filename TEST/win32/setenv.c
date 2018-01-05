
#include <stdlib.h>

#include "setenv.h"

int setenv(const char *name, const char *value, int overwrite)
{
    if(overwrite != 0 || getenv(name) != NULL)
    {
        return _putenv_s(name, value) == 0 ? 0 : -1;
    }
    return 0;
}