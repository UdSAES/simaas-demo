within DemoCloudNativeSIMaaS.Records.Data;
record TempCompCircuitFromMat "Read parameter set for temperature compensation circuit from .mat-file"
  extends DemoCloudNativeSIMaaS.Records.Base.TempCompCircuit(
    r1=scalar(DataFiles.readMATmatrix(fileName, "R1")),
    r2=scalar(DataFiles.readMATmatrix(fileName, "R2")),
    r3=scalar(DataFiles.readMATmatrix(fileName, "R3")),
    r4=scalar(DataFiles.readMATmatrix(fileName, "R4")),
    th1=scalar(DataFiles.readMATmatrix(fileName, "TH1")),
    b1=scalar(DataFiles.readMATmatrix(fileName, "B1")),
    th2=scalar(DataFiles.readMATmatrix(fileName, "TH2")),
    b2=scalar(DataFiles.readMATmatrix(fileName, "B2")));
end TempCompCircuitFromMat;
