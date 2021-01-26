within DemoCloudNativeSIMaaS.Examples;
model TemperatureCompensation
  "Temperature compensation circuit using thermistors"
  Circuits.ThermistorBridge thermistorBridge(redeclare Records.Data.SolutionEDN data)
    annotation (Placement(transformation(extent={{10,-10},{30,10}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature prescribedTemperature
    annotation (Placement(transformation(extent={{-44,-4},{-36,4}})));
  Modelica.Blocks.Math.UnitConversions.From_degC from_degC
    annotation (Placement(transformation(extent={{-58,-4},{-50,4}})));
  Modelica.Blocks.Sources.Ramp ramp(
    height=125,
    duration=60,
    offset=-40,
    startTime=0) annotation (Placement(transformation(extent={{-84,-4},{-76,4}})));
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
  Components.Polynomial2 ref(
    a=1.125e-5,
    b=-1.125e-4,
    c=1.026e-1) annotation (Placement(transformation(extent={{-4,-64},{4,-56}})));
equation
  connect(from_degC.y, prescribedTemperature.T) annotation (Line(points={{-49.6,0},{-44.8,0}}, color={0,0,127}));
  connect(prescribedTemperature.port, thermistorBridge.heatPort1)
    annotation (Line(points={{-36,0},{10,0}}, color={191,0,0}));
  connect(ramp.y, from_degC.u) annotation (Line(points={{-75.6,0},{-58.8,0}}, color={0,0,127}));
  connect(constantVoltage.p, thermistorBridge.p1)
    annotation (Line(points={{-10,8},{-10,20},{20,20},{20,10}}, color={0,0,255}));
  connect(constantVoltage.n, thermistorBridge.n1)
    annotation (Line(points={{-10,-8},{-10,-20},{20,-20},{20,-10}}, color={0,0,255}));
  connect(ground.p, constantVoltage.n) annotation (Line(points={{-10,-28},{-10,-8}}, color={0,0,255}));
  connect(voltageSensor.p, thermistorBridge.p2) annotation (Line(points={{40,-4},{40,0},{30,0}}, color={0,0,255}));
  connect(voltageSensor.n, thermistorBridge.n1)
    annotation (Line(points={{40,-16},{40,-20},{20,-20},{20,-10}}, color={0,0,255}));
  connect(ramp.y, ref.u) annotation (Line(points={{-75.6,0},{-66,0},{-66,-60},{-4.8,-60}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(StopTime=60, __Dymola_Algorithm="Dassl"),
    __Dymola_Commands(file="Scripts/plotExamplesTemperatureCompensation.mos" "plotResult"));
end TemperatureCompensation;
