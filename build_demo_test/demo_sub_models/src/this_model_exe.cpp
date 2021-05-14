#include "this_model_exe.h"

int NetExecutor::Init(string configName)
{
    printf("sub0 %s  %d\r\n",__FUNCTION__,__LINE__);
    return 0;
}

int NetExecutor::Exec(vector<shared_ptr<HiTensor>> &input_vec,
                      vector<shared_ptr<HiTensor>> &output_vec)
{
    printf("sub0 %s  %d\r\n",__FUNCTION__,__LINE__);
    return 0;
}

int NetExecutor::Destory()
{
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    return 0;
}

Executor* CreateNetExecutor()
{
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
    return new NetExecutor();
}
