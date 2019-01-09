#include <iostream>
#include <algorithm> 
#include <fstream>
#include <string.h>
#include <math.h>
#include <unordered_map>
#include <climits>
#include <vector>
#include <assert.h>
using namespace std;

int iterations = 0;
int totalNumberofNodes = 0;
int numAttributes = 0;
vector<string> listAttributes;
vector<int> typeAttribute;	/// 0 : Continuous,	1 : Discrete
vector<pair<vector<int>,int>> allExamples;
vector<int> listLabels;
vector<pair<vector<int>,int>> valExamples;

void input(char filename[]){
	ifstream in;
	in.open(filename);
	if (!in.is_open()){
		cout << "File not found.\n";
	}
	else{
		in >> numAttributes;
		for (int i = 0; i<numAttributes; i++){
			int type = 0;
			string a = "";
			in >> type >> a;
			listAttributes.push_back(a);
			typeAttribute.push_back(type);
		}
		string s;
		while (true){
			vector<int> v;
			int a = 0;
			in >> a;
			if (in.eof()) break;
			v.push_back(a);
			for (int i = 1; i < numAttributes-1; i++){
				in >> a;
				v.push_back(a);
			}
			in >> a;
			if (find(listLabels.begin(),listLabels.end(),a) == listLabels.end())
				listLabels.push_back(a);
			allExamples.push_back(make_pair(v,a));
		}
	}
}

int getMedian(vector<int> &examples, int att){
	vector<int> v;
	for (int i = 0; i<examples.size(); i++){
		v.push_back((allExamples[examples[i]].first)[att]);
	}
	int n = v.size();
	if (n%2 == 0){
		nth_element(v.begin(),v.begin()+(n/2)-1,v.end());
		return v[n/2 - 1];
	}
	else{
		nth_element(v.begin(),v.begin()+(n/2),v.end());
		return v[n/2];
	}	
}

int getIndex(vector<int> &v, int c){
	for (int i = 0; i<v.size(); i++){
		if (v[i]==c)
			return i;
	}
	return -1;
}

struct Node{
	Node* left;
	Node* right;
	bool isLeaf;
	int label;
	int chosenAttribute;
	int splitValue;
	vector<int> examples;
	vector<int> attributes;
	int correct;
	int possiblyCorrect;

	Node(Node* a, Node* a1, bool b, int c, int d, int e, vector<int> v1, vector<int> v2, int p1, int p2){
		this->left = a;
		this->right = a1;
		this->isLeaf = b;
		this->label = c;
		this->chosenAttribute = d;
		this->splitValue = e;
		this->examples = v1;
		this->attributes = v2;
		this->correct = p1;
		this->possiblyCorrect = p2;
	}
	int getBestLabel();
	int getBestAttribute();
};

int Node::getBestLabel(){
	// find the label with max occurrences
	vector<int> lLabels;
	for (int i=0; i<(listLabels.size()); i++){
		lLabels.push_back(0);
	}	 
	for (int i = 0; i<this->examples.size(); i++){
		int index = getIndex(listLabels,(allExamples[this->examples[i]]).second);
		lLabels[index]++;
	}
	int max = INT_MIN;
	int maxLabel = -1;
	for (int i=0; i<listLabels.size(); i++){
		if (lLabels[i] >= max){
			max = lLabels[i];
			maxLabel = listLabels[i];
		}
	}
	return maxLabel;
}

int Node::getBestAttribute(){
	// Find the attribute with minimum entropy of H(Y/Xj)
	float min = float(INT_MAX);
	float minAtt = -1;
	vector<int> leftLabels;
	vector<int> rightLabels;
	for (int i=0; i<listLabels.size(); i++){
		leftLabels.push_back(0);
		rightLabels.push_back(0);
	}
	vector<float> lList;
	for (int i = 0; i<this->attributes.size(); i++){
		for (int i=0; i<listLabels.size(); i++){
			leftLabels[i] = 0;
			rightLabels[i] = 0;
		}	
		int att = this->attributes[i];
		if (typeAttribute[att] == 0){	/// Continuous
			int median = getMedian(this->examples,att);
			for (int j=0; j<this->examples.size(); j++){
				if ((allExamples[this->examples[j]].first)[att] <= median){	////Left
					int indexLabel = getIndex(listLabels,(allExamples[this->examples[j]]).second);
					leftLabels[indexLabel] ++;
				}
				else{	/////Right
					int indexLabel = getIndex(listLabels,(allExamples[this->examples[j]]).second);
					rightLabels[indexLabel] ++;
				}
			}
		}
		else{		/// Discrete
			for (int j=0; j<this->examples.size(); j++){
				if ((allExamples[this->examples[j]].first)[att] == 0){	////Left
					int indexLabel = getIndex(listLabels,(allExamples[this->examples[j]]).second);
					leftLabels[indexLabel] ++;
				}
				else{	/////Right
					int indexLabel = getIndex(listLabels,(allExamples[this->examples[j]]).second);
					rightLabels[indexLabel] ++;
				}
			}
		}
		int leftSum = accumulate(leftLabels.begin(), leftLabels.end(), 0);
		int rightSum = accumulate(rightLabels.begin(), rightLabels.end(), 0);
		float lhyx = 0.0;
		for (int i=0; i<leftLabels.size(); i++){
			float v = float(leftLabels[i])/float(leftSum);
			if (v > 0.0001)
				lhyx -= v * log2(v);
		}
		float l = (float(leftSum)/float(leftSum+rightSum))*lhyx;
		float rhyx = 0.0;
		for (int i=0; i<rightLabels.size(); i++){
			float v = float(rightLabels[i])/float(rightSum);
			if (v> 0.0001)
				rhyx -= v * log2(v);
		}
		float r = (float(rightSum)/float(leftSum+rightSum))*rhyx;
		lList.push_back((l+r));
		if (min > (l+r)){
			min = (l+r);
			minAtt = att;
		}
	}
	for (int i =0 ;i<lList.size();i++){
		if (lList[i]!=min)
			return minAtt;
	}
	return -1;
}

Node* root;
ofstream outputfile;

int getLabel(Node* tree, vector<int> example){
	if (tree->isLeaf){
		return tree->label;
	}	
	else{
		if (typeAttribute[tree->chosenAttribute] == 0){			///Continuous
			if (example[tree->chosenAttribute] <= tree->splitValue){
				return getLabel(tree->left,example);
			}
			else{
				return getLabel(tree->right,example);
			}
		}
		else{					///Discrete
			if (example[tree->chosenAttribute] == 0){
				return getLabel(tree->left,example);
			}
			else{
				return getLabel(tree->right,example);
			}
		}
	}
}

float checkAccuracy(char filename[]){
	ifstream in;
	in.open(filename);
	if (!in.is_open()){
		cout << "File not found.\n";
	}
	else{
		in >> numAttributes;
		for (int i = 0; i<numAttributes; i++){
			int type = 0;
			string a = "";
			in >> type >> a;
		}
		int correctCount = 0;
		int totalCount = 0;
		string s;
		while (true){
			vector<int> v;
			int a = 0;
			in >> a;
			if (in.eof()) break;
			v.push_back(a);
			for (int i = 1; i < numAttributes-1; i++){
				in >> a;
				v.push_back(a);
			}
			in >> a;
			/// a contains the label and v contains the example
			int la = getLabel(root,v);
			// cout << "Predicted : " << la << " | Actual : " << a << endl;
			if (la == a)
				correctCount ++;
			totalCount++;
		}
		return ((float(correctCount)/float(totalCount)) * 100); 
	}
}

void plot(char filename1[], char filename2[], char filename3[]){
	float c1 = checkAccuracy(filename1);
	float c2 = checkAccuracy(filename2);
	float c3 = checkAccuracy(filename3);
	string str = "";
	str = to_string(totalNumberofNodes) + "," + to_string(c1) + "," + to_string(c2) + "," + to_string(c3) + "\n";
	outputfile << str;
}

void growTree(Node *node, char filename1[], char filename2[], char filename3[]){
	totalNumberofNodes++;
	bool b = true;
	int y = (allExamples[node->examples[0]]).second;
	for (int i = 0; i<node->examples.size(); i++){
		if ((allExamples[node->examples[i]]).second != y){
			b = false;
			break; 
		}
	}
	if (b){	/// Leaf
		node->label = y;
		node->isLeaf = true;
		// plot(filename1,filename2,filename3);
	}
	else if (node->attributes.size() == 0){
		node->label = node->getBestLabel();
		node->isLeaf = true;
		// plot(filename1,filename2,filename3);
	}
	else{	//Grow
		int att = node->getBestAttribute();
		if (att == -1){
			node->label = node->getBestLabel();
			node->isLeaf = true;
			// Check accuracy
			// plot(filename1,filename2,filename3);
			return;
		}
		else{
			node->chosenAttribute = att;
			node->label = node->getBestLabel();
			node->isLeaf = true;
			// Check accuracy
			// plot(filename1,filename2,filename3);
			node->isLeaf = false;
			vector<int> leftEx;
			vector<int> rightEx;
			vector<int> atts;	
			if (typeAttribute[att] == 0){	///Continuous
				int median = getMedian(node->examples, att);
				node->splitValue = median;
				for (int i = 0; i<node->examples.size(); i++){
					if ((allExamples[node->examples[i]].first)[att] <= median)
						leftEx.push_back(node->examples[i]);
					else
						rightEx.push_back(node->examples[i]);
				}
				// cout << "Continuous LeftSize : " << leftEx.size() << endl;
				// cout << "Continuous RightSize : " << rightEx.size() << endl;
				for (int i =0; i<node->attributes.size(); i++){
					atts.push_back(node->attributes[i]);
				}
				node->left = new Node(NULL,NULL,false,-1,-1,-1,leftEx,atts,0,0);
				growTree(node->left,filename1,filename2,filename3);
				node->right = new Node(NULL,NULL,false,-1,-1,-1,rightEx,atts,0,0);
				growTree(node->right,filename1,filename2,filename3);
			}
			else{	/// Discrete
				for (int i = 0; i<node->examples.size(); i++){
					if ((allExamples[node->examples[i]].first)[att] == 0)
						leftEx.push_back(node->examples[i]);
					else
						rightEx.push_back(node->examples[i]);
				}
				// cout << "Discrete LeftSize : " << leftEx.size() << endl;
				// cout << "Discrete RightSize : " << rightEx.size() << endl;
				for (int i =0; i<node->attributes.size(); i++){
					if (att != node->attributes[i])	
						atts.push_back(node->attributes[i]);
				}
				node->left = new Node(NULL,NULL,false,-1,-1,-1,leftEx,atts,0,0);
				growTree(node->left,filename1,filename2,filename3);
				node->right = new Node(NULL,NULL,false,-1,-1,-1,rightEx,atts,0,0);
				growTree(node->right,filename1,filename2,filename3);
			}
		}	
	}
	return;
}

int pruneTree(Node *node, vector<int> examples){
	int correctCount = 0;
	int totalCount = 0;
	for (int i = 0; i<examples.size(); i++){
		if (valExamples[examples[i]].second == node->label)
			correctCount++;
		totalCount++;
	}
	int acc1 = correctCount;
	if (node->isLeaf){
		return acc1;
	}
	else{
		int att = node->chosenAttribute;
		vector<int> leftEx;
		vector<int> rightEx;
		if (typeAttribute[att] == 0){	///Continuous
			int median = node->splitValue;
			for (int i = 0; i<examples.size(); i++){
				if ((valExamples[examples[i]].first)[att] <= median)
					leftEx.push_back(examples[i]);
				else
					rightEx.push_back(examples[i]);
			}
		}
		else{	/// Discrete
			for (int i = 0; i<examples.size(); i++){
				if ((valExamples[examples[i]].first)[att] == 0)
					leftEx.push_back(examples[i]);
				else
					rightEx.push_back(examples[i]);
			}
		}
		int lacc = pruneTree(node->left, leftEx);
		int racc = pruneTree(node->right, rightEx);
		int acc2 = lacc + racc;
		if (acc1<=acc2){
			// Don't prune
			return acc2;
		}
		else{
			// Prune
			node->isLeaf = true;
			delete(node->left);
			delete(node->right);
			return acc1;
		}
	}
}

bool moveDownTree(Node *node, int i){
	if (!node->isLeaf){
		if (node->label == valExamples[i].second)
			node->possiblyCorrect++;
		if (typeAttribute[node->chosenAttribute]==0){
			if ((valExamples[i].first)[node->chosenAttribute] <= node->splitValue){
				if (moveDownTree(node->left,i)){
					node->correct++;
					return true;
				}
			}
			else{
				if (moveDownTree(node->right,i)){
					node->correct++;
					return true;
				}
			}
		}
		else{
			if ((valExamples[i].first)[node->chosenAttribute] == 0){
				if (moveDownTree(node->left,i)){
					node->correct++;
					return true;
				}
			}
			else{
				if (moveDownTree(node->right,i)){
					node->correct++;
					return true;
				}
			}
		}
		return false;
	}
	else{
		if (node->label == valExamples[i].second){
			node->correct++;
			return true;
		}
		else
			return false;
	}	
}

pair<int,Node*> getMaxNode(Node *node){
	if (node->isLeaf){
		return (make_pair((node->possiblyCorrect)-(node->correct),node));
	}
	else{
		int n1 = (node->possiblyCorrect)-(node->correct);
		pair<int,Node*> left = getMaxNode(node->left);
		pair<int,Node*> right = getMaxNode(node->right);
		int n2 = left.first;
		int n3 = right.first;
		if (n1>=n2 && n1>=n3){
			return make_pair((node->possiblyCorrect)-(node->correct),node);
		}
		else if (n2>=n1 && n2>=n3)
			return left;
		else
			return right;
	}
}

bool pruneNodeinTree(Node *node){
	// Returns true if there is a node with diff > 1 and also nulls its kid
	pair<int,Node*> max = getMaxNode(node);
	if (max.first > 1){
		(max.second)->isLeaf = true;
		delete((max.second)->left);
		delete((max.second)->right);
		return true;
	}
	return false;
}

void setToZero(Node *node){
	node->correct = 0;
	node->possiblyCorrect = 0;
	if (!node->isLeaf){
		setToZero(node->left);
		setToZero(node->right);	
	}
}

bool traverseTree(Node *node, vector<int> examples){
	for (int i = 0; i<examples.size(); i++){
		moveDownTree(node,examples[i]);
	}
	return pruneNodeinTree(node);
}

int pruneTree2(Node* tree, vector<int> examples){
	while (traverseTree(tree,examples)){
		iterations++;
		setToZero(tree);
	}
	return 0;
}

int pruneTree3(Node* tree, vector<int> examples, char filename1[], char filename2[], char filename3[]){
	while (traverseTree(tree,examples)){
		plot(filename1,filename2,filename3);
		iterations++;
		setToZero(tree);
	}
	return 0;
}

void postPrune(char filename[], Node *node){
	ifstream in;
	in.open(filename);
	if (!in.is_open()){
		cout << "File not found.\n";
	}
	else{
		in >> numAttributes;
		for (int i = 0; i<numAttributes; i++){
			int type = 0;
			string a = "";
			in >> type >> a;
		}
		string s;
		vector<int> examples;
		int counter = 0;
		while (true){
			vector<int> v;
			int a = 0;
			in >> a;
			if (in.eof()) break;
			v.push_back(a);
			for (int i = 1; i < numAttributes-1; i++){
				in >> a;
				v.push_back(a);
			}
			in >> a;
			examples.push_back(counter);
			valExamples.push_back(make_pair(v,a));
			counter++;
		}
		int acc = pruneTree(node, examples);
	}
}

void checkAccuracy(char filename[], Node* tree){
	ifstream in;
	in.open(filename);
	if (!in.is_open()){
		cout << "File not found.\n";
	}
	else{
		in >> numAttributes;
		for (int i = 0; i<numAttributes; i++){
			int type = 0;
			string a = "";
			in >> type >> a;
		}
		int correctCount = 0;
		int totalCount = 0;
		string s;
		while (true){
			vector<int> v;
			int a = 0;
			in >> a;
			if (in.eof()) break;
			v.push_back(a);
			for (int i = 1; i < numAttributes-1; i++){
				in >> a;
				v.push_back(a);
			}
			in >> a;
			/// a contains the label and v contains the example
			int la = getLabel(tree,v);
			// cout << "Predicted : " << la << " | Actual : " << a << endl;
			if (la == a)
				correctCount ++;
			totalCount++;
		}
		cout << "Correct/Total : " << to_string(correctCount) << "/" << to_string(totalCount) << endl;
		cout << "Accuracy : " << to_string((float(correctCount)/float(totalCount)) * 100) << '%' << endl; 
	}
}

int main(int argc, char * argv[]){

	srand(time(NULL));
	input(argv[1]);
	vector<int> atts;
	for (int i = 0; i<numAttributes-1; i++){
		atts.push_back(i);
	}
	vector<int> exs;
	for (int i = 0; i<allExamples.size(); i++){
		exs.push_back(i);
	}
	root = new Node(NULL,NULL,false,-1,-1,-1,exs,atts,0,0);	
	
	cout << "--------------------------------------" << endl;
	cout << "Starting to grow the tree..." << endl;
	
	outputfile.open("growAcc.csv");
	growTree(root,argv[1],argv[2],argv[3]);
	outputfile.close();
	
	cout << "Done!" << endl;
	cout << "Total Nodes in tree : " << totalNumberofNodes << endl; 
	cout << "\n--------------------------------------" << endl;
	
	cout << "Before pruning :" << endl;
	cout << "\nOn training data : " << endl;
	checkAccuracy(argv[1],root);
	cout << "\nOn validation data : " << endl;
	checkAccuracy(argv[2],root);
	cout << "\nOn testing data : " << endl;
	checkAccuracy(argv[3],root);
	cout << "\n--------------------------------------" << endl;
	
	cout << "Starting to prune the tree w.r.t Validation Data..." << endl;
	
	outputfile.open("pruneAcc.csv");
	postPrune(argv[2],root);
	outputfile.close();
	
	cout << "Number of iterations : " << iterations << endl;
	cout << "Done!" << endl;
	cout << "\n--------------------------------------" << endl;
	
	cout << "After pruning :" << endl;
	cout << "\nOn training data : " << endl;
	checkAccuracy(argv[1],root);
	cout << "\nOn validation data : " << endl;
	checkAccuracy(argv[2],root);
	cout << "\nOn testing data : " << endl;
	checkAccuracy(argv[3],root);
	cout << "--------------------------------------" << endl;
	cout << endl;
	return 0;
}