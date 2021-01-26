within DemoCloudNativeSIMaaS.ModelExport;
model TemperatureCompensationFMU "FMU for identification of thermistor-network component values"
  Circuits.ThermistorBridge thermistorBridge(redeclare Records.Data.TempCompCircuitFromMat data)
            annotation (Placement(transformation(extent={{10,-10},{30,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature prescribedTemperature
    annotation (Placement(transformation(extent={{-44,-4},{-36,4}})));
  Modelica.Blocks.Math.UnitConversions.From_degC from_degC
    annotation (Placement(transformation(extent={{-58,-4},{-50,4}})));
  Modelica.Electrical.Analog.Sources.ConstantVoltage constantVoltage(V=1.1)
    annotation (Placement(transformation(
        extent={{-8,8},{8,-8}},
        rotation=270,
        origin={-10,0})));
  Modelica.Electrical.Analog.Basic.Ground ground annotation (Placement(transformation(extent={{-16,-40},{-4,-28}})));
  Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor
    annotation (Placement(transformation(
        extent={{6,-6},{-6,6}},
        rotation=90,
        origin={40,-10})));
  Modelica.Blocks.Interfaces.RealInput temperature(unit="degC") "Temperature of the circuit's components"
    annotation (Placement(transformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealOutput voltage(unit="V") "Voltage at point B as output signal"
    annotation (Placement(transformation(extent={{90,-10},{110,10}})));
equation
  connect(from_degC.y,prescribedTemperature. T) annotation (Line(points={{-49.6,0},{-44.8,0}}, color={0,0,127}));
  connect(prescribedTemperature.port,thermistorBridge. heatPort1)
    annotation (Line(points={{-36,0},{10,0}}, color={191,0,0}));
  connect(constantVoltage.p,thermistorBridge. p1)
    annotation (Line(points={{-10,8},{-10,20},{20,20},{20,10}}, color={0,0,255}));
  connect(constantVoltage.n,thermistorBridge. n1)
    annotation (Line(points={{-10,-8},{-10,-20},{20,-20},{20,-10}}, color={0,0,255}));
  connect(ground.p,constantVoltage. n) annotation (Line(points={{-10,-28},{-10,-8}}, color={0,0,255}));
  connect(voltageSensor.p,thermistorBridge. p2) annotation (Line(points={{40,-4},{40,0},{30,0}}, color={0,0,255}));
  connect(voltageSensor.n,thermistorBridge. n1)
    annotation (Line(points={{40,-16},{40,-20},{20,-20},{20,-10}}, color={0,0,255}));
  connect(from_degC.u, temperature) annotation (Line(points={{-58.8,0},{-100,0}}, color={0,0,127}));
  connect(voltageSensor.v, voltage) annotation (Line(points={{46.6,-10},{60,-10},{60,0},{100,0}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(coordinateSystem(preserveAspectRatio=false)));
end TemperatureCompensationFMU;
