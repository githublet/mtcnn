#include <io.h>
#include <stdio.h>
#include <fstream>
#include <caffe.pb.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iosfwd>
#include <iostream>
#include <map>
#include <string>

#ifndef O_BINARY

#define O_BINARY 0 

#endif

using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::Message;

using namespace std;

struct info{
	int kernel_size;
	int num_output;
	int isfc;

	info(){ memset(this, 0, sizeof(*this)); }
	info(int kernelsize, int numoutput) :kernel_size(kernelsize), num_output(numoutput){}
};

int pos = 0;
void writeHand(FILE* f, const char* name){
	fprintf(f, "//--------------------------%s---------------------------------\nstatic const float model_weights_%s_[] = {", name, name);
	pos = 0;
}

void writeData(FILE* f, const float* data, int len)
{
	for (int i = 0; i < len; ++i, ++pos)
	{
		/*
        if (abs(data[i]) < 1e-04)
		{
			if (data[i]<0)
			{
				fprintf(f, "[ %.8e]", data[i]);
				fprintf(f, "\n");
			}
			else
			{
				fprintf(f, "[  %.8e]", data[i]);
				fprintf(f, "\n");
			}
		}
		else
		{
			if (data[i] < 0)
			{
				fprintf(f, "[%.8f]", data[i]);
				fprintf(f, "\n");
			}
			else
			{
				fprintf(f, "[ %.8f]", data[i]);
				fprintf(f, "\n");
			}
		}
        */
        fprintf(f, "%.8f,\n", data[i]);
		//fprintf(f, "[%.8f]", data[i]);
		//fprintf(f, "\n");
	}
}

void writeEnd(FILE* f)
{
	fprintf(f, "};\n\n\n");
}

bool loadDep(const char* file, Message* net){
	int fd = open(file, O_RDONLY);
	if (fd == -1) return false;

	FileInputStream* input = new FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, net);
	delete input;
	close(fd);
	return success;
}

bool loadCaffemodel(const char* file, Message* net){
	int fd = open(file, O_RDONLY | O_BINARY);
	if (fd == -1) return false;

	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	bool success = net->ParseFromCodedStream(coded_input);
	delete coded_input;
	delete raw_input;
	close(fd);
	return success;
}
//int main()
//{
//	float a = 1e-05;
//	printf("%.8f", a);
//	cin.get();
//	return 0;
//}

//这个程序是产生mtcnn模型头文件的
int main(){
	//12 P
	//24 R
	//48 O
	vector<string> names = { "PNet.txt", "RNet.txt", "ONet.txt" };

	//注意这里的3个模型不支持原生mtcnn训练的模型，因为原生模型是matlab训练的，有转置，所以直接套用到mtcnn-light时会无效
	const char* pnet = "det1.caffemodel";
	const char* rnet = "det2.caffemodel";
	const char* onet = "det3.caffemodel";

	vector<string> caffemodel = {pnet, rnet, onet};
	FILE* fmodel = NULL;
	/*fprintf(fmodel, 
		"#ifndef MTCNN_MODELS_H\n"
		"#define MTCNN_MODELS_H\n"
		"\n\n\n");*/
	for (int i = 0; i < caffemodel.size(); ++i)
	{
		fmodel = fopen(names[i].c_str(), "wb");
		//writeHand(fmodel, names[i].c_str());
		printf("=======================%s===========================\n", names[i].c_str());
		NetParameter net;
		bool success = loadCaffemodel(caffemodel[i].c_str(), &net);
		if (!success)
		{
			printf("读取错误啦:%s\n", caffemodel[i].c_str());
			return 0;
		}

		for (int i = 0; i < net.layer_size(); ++i)
		{
			LayerParameter& param = *net.mutable_layer(i);
			int n = param.mutable_blobs()->size();
			if (n)
			{
				const BlobProto& blob = param.blobs(0);
				printf("layer: %s weight(%d)", param.name().c_str(), blob.data_size());
				writeData(fmodel, blob.data().data(), blob.data_size());

				if (n > 1)
				{
					const BlobProto& bais = param.blobs(1);
					printf(" bais(%d)", bais.data_size());
					writeData(fmodel, bais.data().data(), bais.data_size());
				}
				printf("\n");
			}
		}
		//writeEnd(fmodel);
		//fprintf(fmodel, "#endif //MTCNN_MODELS_H");
		fclose(fmodel);
	}
	
}
