within DemoCloudNativeSIMaaS.Records.Base;
partial record TempCompCircuit "The parameters of the temperature compensation circuit"
  extends Modelica.Icons.Record;

  parameter String fileName="noFile" "Filepath to external file storing actual data";

  parameter Modelica.SIunits.Resistance r1 "Resistance at temperature T_ref; R1";
  parameter Modelica.SIunits.Resistance r2 "Resistance at temperature T_ref; R2";
  parameter Modelica.SIunits.Resistance r3 "Resistance at temperature T_ref; R3";
  parameter Modelica.SIunits.Resistance r4 "Resistance at temperature T_ref; R4";
  parameter Modelica.SIunits.Resistance th1 "Resistance at temperature T_ref; TH1";
  parameter Modelica.SIunits.Temp_K b1 "B-parameter / 1/K; TH1";
  parameter Modelica.SIunits.Resistance th2 "Resistance at temperature T_ref; TH2";
  parameter Modelica.SIunits.Temp_K b2 "B-parameter / 1/K; TH2";

end TempCompCircuit;
