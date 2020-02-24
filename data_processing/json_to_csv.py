import os
import csv

def json_to_csv(json_dir, result_path):

    filepath = json_dir
    result_file_path = result_path
    allfiles = os.listdir(filepath)
    
    with open(result_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename'.ljust(40), 'Difficult Case'.ljust(20), 'Abnormal'.ljust(20), 'Confirmed'.ljust(20),'label'.ljust(20)])
        writer.writerow([])
    
    
    for afile in allfiles:
        filedir = os.path.join(filepath, afile)
        Exclude = os.popen("grep Difficult "+filedir)
        Exclude = Exclude.readline()
        if Exclude:
            Exclude = "true" if Exclude.split(":")[-1][1]=="t" else "false"
    
        DxUnclear = os.popen("grep Abnormal "+filedir)
        DxUnclear = DxUnclear.readline()
        if DxUnclear:
            DxUnclear = "true" if DxUnclear.split(":")[-1][1]=="t" else "false"
    
        Confirmed = os.popen("grep Confirmed "+filedir)
        Confirmed = Confirmed.readline()
        if Confirmed:
            Confirmed = "true" if Confirmed.split(":")[-1][1]=="t" else "false"
    
        label = os.popen("grep label "+ filedir)
        label = label.readline()
        if label:
            label = label.split(":")[-1]
            label = label[label.find("\"")+1:label.find(",")-1]
        print(label)
        filename = filedir.split("/")[-1]
        with open(result_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if label:
                writer.writerow([filename.ljust(40), Exclude.ljust(20), DxUnclear.ljust(20), Confirmed.ljust(20), label.ljust(20)])
            else:
                writer.writerow([filename.ljust(40), Exclude.ljust(20), DxUnclear.ljust(20), Confirmed.ljust(20)])
def main():
    json_to_csv("./PXR_V1V2_Annotations", "hipfx_label.csv")

if __name__ == "__main__":
        main()
