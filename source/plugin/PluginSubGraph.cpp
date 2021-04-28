#include "MNN/plugin/PluginKernel.hpp"
#include "MNN/plugin/PluginShapeInference.hpp"
#include <iostream>
#include "core/TensorUtils.hpp"
#include "plugin/executor.h"
#include <dlfcn.h>
#include <fstream> 

MNN_PUBLIC int _intSubGraph = 10; // Just for linking successfully.

namespace MNN {
namespace plugin {

static std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> res;
    if("" == str) return res;

    char * strs = new char[str.length() + 1] ; 
    strcpy(strs, str.c_str()); 
 
    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());
 
    char *p = strtok(strs, d);
    while(p) {
        std::string s = p;
        res.push_back(s);
        p = strtok(NULL, d);
    }
 
    return res;
}

namespace shape_inference {
class SubGraph : public InferShapeKernel {
public:
    bool compute(InferShapeContext* ctx) override;
};

bool SubGraph::compute(InferShapeContext* ctx) {
    //只需要计算输出shape
    std::string inputs_info, outputs_info;
    if (ctx->hasAttr("inputs_info")) {
        inputs_info  = ctx->getAttr("inputs_info")->s()->str();
    }
    if (ctx->hasAttr("outputs_info")) {
        outputs_info = ctx->getAttr("outputs_info")->s()->str();
    }
    std::string input_index_map, output_index_map;
    if (ctx->hasAttr("input_index_map")) {
        input_index_map  = ctx->getAttr("input_index_map")->s()->str();
    }
    if (ctx->hasAttr("output_index_map")) {
        output_index_map = ctx->getAttr("output_index_map")->s()->str();
    }
    //395:0;394:1;
    std::map<std::string, int> o_idx_mp;
    for(auto a: split(output_index_map, ";"))
    {
        auto b =  split(a, ":");
        o_idx_mp.insert(std::pair<std::string, int>(b[0], std::stoi(b[1])));
    }

    //394:float:NCHW:1,58,28,28
    for(auto a: split(outputs_info, ";"))
    {
        auto b =  split(a, ":");
        std::string name    = b[0];
        int         index   = o_idx_mp[name];
        std::string type    = b[1];
        std::string format  = b[2];

        auto c =  split(b[3], ",");
        auto& output  = ctx->output(index)->buffer();
        output.dimensions    = c.size();
        output.dim[0].extent = atoi(c[0].c_str());
        output.dim[1].extent = atoi(c[1].c_str());
        output.dim[2].extent = atoi(c[2].c_str());
        output.dim[3].extent = atoi(c[3].c_str());
        assert(type == "float");
        output.type = halide_type_of<float>();

        TensorUtils::getDescribe(ctx->output(index))->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    }

    return true /*success*/;
}
} // namespace shape_inference


namespace backend {
class SubGraph : public CPUComputeKernel {
  private:
    Executor* net_executor_;
    vector<shared_ptr<HiTensor>> input_vec_;
    vector<shared_ptr<HiTensor>> output_vec_;
  public:
    bool init(CPUKernelContext*) override;
    bool compute(CPUKernelContext* ctx) override;
};

bool connetion_input_to_hitensor(CPUKernelContext* ctx,
                                 vector<shared_ptr<HiTensor>> &input_vec,
                                 vector<shared_ptr<HiTensor>> &output_vec)  {

    //从属性中获取MNN上下文的输入输出信息
    std::string inputs_info, outputs_info;
    if (ctx->hasAttr("inputs_info")) {
        inputs_info  = ctx->getAttr("inputs_info")->s()->str();
    }
    if (ctx->hasAttr("outputs_info")) {
        outputs_info = ctx->getAttr("outputs_info")->s()->str();
    }
    std::string input_index_map, output_index_map;
    if (ctx->hasAttr("input_index_map")) {
        input_index_map  = ctx->getAttr("input_index_map")->s()->str();
    }
    if (ctx->hasAttr("output_index_map")) {
        output_index_map = ctx->getAttr("output_index_map")->s()->str();
    }
    //获取MNN上下文已经申请好IO内存
    int I = ctx->inputs().size();
    float** inputs = new float*[I];
    for (int i = 0; i < I; i++) {
        inputs[i] = reinterpret_cast<float*>(ctx->input(i)->buffer().host);
    }
    int O = ctx->outputs().size();
    float** outputs = new float*[O];
    for (int i = 0; i < O; i++) {
        outputs[i] = reinterpret_cast<float*>(ctx->output(i)->buffer().host);
    }

    //将MNN上下文的IO 与NPU的io vector对应起来
    //====================输入的对应====================
    std::map<std::string, int> i_idx_mp;
    for(auto a: split(input_index_map, ";"))
    {
        auto b =  split(a, ":");
        i_idx_mp.insert(std::pair<std::string, int>(b[0], std::stoi(b[1])));
    }
    for(auto a: split(inputs_info, ";"))
    {
        auto b =  split(a, ":");
        std::string name    = b[0];
        int         index   = i_idx_mp[name];
        std::string type    = b[1];
        std::string format  = b[2];

        auto c =  split(b[3], ",");

        shared_ptr<HiTensor> in_tensor = shared_ptr<HiTensor>(new HiTensor);
        in_tensor->tensor_name = name;
        in_tensor->index       = index;
        in_tensor->data        = inputs[index];
        assert(type == "float");//目前只支持float的tensor
        in_tensor->data_type   = HI_ACL_FLOAT;
        if (format == "NCHW") 
        {
            in_tensor->data_format = HI_ACL_FORMAT_NCHW;
            in_tensor->n  = atoi(c[0].c_str());
            in_tensor->c  = atoi(c[1].c_str());
            in_tensor->h  = atoi(c[2].c_str());
            in_tensor->w  = atoi(c[3].c_str());   
        }
        else if (format == "NHWC")
        {
            in_tensor->data_format = HI_ACL_FORMAT_NHWC;
            in_tensor->n  = atoi(c[0].c_str());
            in_tensor->h  = atoi(c[1].c_str());
            in_tensor->w  = atoi(c[2].c_str());
            in_tensor->c  = atoi(c[3].c_str());   
        }
        else
        {
            assert(false);//目前只支持nchw和nhwc
        }
        input_vec.push_back(in_tensor);
    }

    //====================输出的对应====================
    std::map<std::string, int> o_idx_mp;
    for(auto a: split(output_index_map, ";"))
    {
        auto b =  split(a, ":");
        o_idx_mp.insert(std::pair<std::string, int>(b[0], std::stoi(b[1])));
    }
    for(auto a: split(outputs_info, ";"))
    {
        auto b =  split(a, ":");
        std::string name    = b[0];
        int         index   = o_idx_mp[name];
        std::string type    = b[1];
        std::string format  = b[2];

        auto c =  split(b[3], ",");

        shared_ptr<HiTensor> out_tensor = shared_ptr<HiTensor>(new HiTensor);
        out_tensor->tensor_name = name;
        out_tensor->index       = index;
        out_tensor->data        = outputs[index];
        assert(type == "float");//目前只支持float的tensor
        out_tensor->data_type   = HI_ACL_FLOAT;  
        if (format == "NCHW") 
        {
            out_tensor->data_format = HI_ACL_FORMAT_NCHW;
            out_tensor->n  = atoi(c[0].c_str());
            out_tensor->c  = atoi(c[1].c_str());
            out_tensor->h  = atoi(c[2].c_str());
            out_tensor->w  = atoi(c[3].c_str());   
        }
        else if (format == "NHWC")
        {
            out_tensor->data_format = HI_ACL_FORMAT_NHWC;
            out_tensor->n  = atoi(c[0].c_str());
            out_tensor->h  = atoi(c[1].c_str());
            out_tensor->w  = atoi(c[2].c_str());
            out_tensor->c  = atoi(c[3].c_str());   
        }
        else
        {
            assert(false);//目前只支持nchw和nhwc
        }
        output_vec.push_back(out_tensor);
    }
    return true;
}
bool SubGraph::init(CPUKernelContext* ctx) {
    std::string conf_file;
    if (ctx->hasAttr("conf_file")) {
        conf_file  = ctx->getAttr("conf_file")->s()->str();
    }
    conf_file = conf_file + ".conf";
    ifstream in(conf_file);  
    string line; 
    if(in) // 有该文件  
    {  
        cout << "Load config..." << endl;
        while (getline(in, line)) // line中不包括每行的换行符  
        {   
            cout << line << endl;  
        }  
    }  
    else // 没有该文件  
    {  
        cout <<"Error no config file:" <<  conf_file << endl;  
    }  
/*
    std::string exe_lib_file, npu_model_file;
    //从配置文件中获取npu执行lib的路径
    void *handle = dlopen(exe_lib_file.c_str(), RTLD_LAZY);
    if(handle == NULL)
    {
        debug_print("error dlopen - %s \r\n", dlerror());
        return false;
    }

    char * dl_error;
    CreatorExec create_net_executor = (CreatorExec)dlsym(handle, "CreateNetExecutor");
    if((dl_error = dlerror()) != NULL)
    {
        debug_print("find sym error %s \r\n", dl_error);
        return false;
    }

    net_executor_ = (*create_net_executor)();//创建执行网络的抓手

    int ret = net_executor_->Init(npu_model_file.c_str());
    if (ret != 0) 
    {
        debug_print("Failed to init net_executor_\r\n");
        return false;
    }
*/
    connetion_input_to_hitensor(ctx, input_vec_, output_vec_);//MNN上下文的IO与net_executor的IO对应起来

    return true;
}

bool SubGraph::compute(CPUKernelContext* ctx) {
    net_executor_->Exec(input_vec_, output_vec_);
    return true;
}

} // namespace backend


REGISTER_PLUGIN_OP(SubGraph, shape_inference::SubGraph);
REGISTER_PLUGIN_COMPUTE_KERNEL(SubGraph, backend::SubGraph);

} // namespace plugin
} // namespace MNN
