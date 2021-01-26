within DemoCloudNativeSIMaaS.Circuits;
model ThermistorBridge "Temperature-dependent H-bridge"

  parameter String fileName = "null" "The path to the .mat-file containing _all_ parameter values";

  Modelica.Electrical.Analog.Basic.Resistor R1(R=data.r1) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=270,
        origin={-20,60})));
  Modelica.Electrical.Analog.Basic.Resistor R2(R=data.r2)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=270,
        origin={0,20})));
  Modelica.Electrical.Analog.Basic.Resistor R3(R=data.r3) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=270,
        origin={-20,-20})));
  Modelica.Electrical.Analog.Basic.Resistor R4(R=data.r4)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=270,
        origin={0,-60})));
  Components.ThermistorNTC TH1(R_ref=data.th1, B=data.b1)
    annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={20,60})));
  Components.ThermistorNTC TH2(R_ref=data.th2, B=data.b2) annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=270,
        origin={20,-20})));
  Modelica.Electrical.Analog.Interfaces.PositivePin p1 "Positive electrical pin"
    annotation (Placement(transformation(extent={{-10,90},{10,110}})));
  Modelica.Electrical.Analog.Interfaces.NegativePin n1 "Negative electrical pin"
    annotation (Placement(transformation(extent={{-10,-110},{10,-90}})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a heatPort1 "Conditional heat port"
    annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
  Modelica.Electrical.Analog.Interfaces.PositivePin p2 "Positive electrical pin"
    annotation (Placement(transformation(extent={{90,-10},{110,10}})));

  replaceable Records.Base.TempCompCircuit data(fileName=fileName)
    annotation (choicesAllMatching=true, Placement(transformation(extent={{-90,70},{-70,90}})));
equation
  connect(R1.p, TH1.p) annotation (Line(points={{-20,70},{-20,80},{20,80},{20,70}}, color={0,0,255}));
  connect(R1.n, TH1.n) annotation (Line(points={{-20,50},{-20,40},{20,40},{20,50}}, color={0,0,255}));
  connect(R2.p, TH1.n) annotation (Line(points={{0,30},{0,40},{20,40},{20,50}}, color={0,0,255}));
  connect(R2.n, R3.p) annotation (Line(points={{0,10},{0,0},{-20,0},{-20,-10}}, color={0,0,255}));
  connect(R2.n, TH2.p) annotation (Line(points={{-1.77636e-15,10},{-1.77636e-15,0},{20,0},{20,-10}}, color={0,0,255}));
  connect(R3.n, TH2.n) annotation (Line(points={{-20,-30},{-20,-40},{20,-40},{20,-30}}, color={0,0,255}));
  connect(R4.p, TH2.n) annotation (Line(points={{0,-50},{0,-40},{20,-40},{20,-30}}, color={0,0,255}));
  connect(R1.p, p1) annotation (Line(points={{-20,70},{-20,80},{0,80},{0,100}}, color={0,0,255}));
  connect(R4.n, n1) annotation (Line(points={{0,-70},{0,-100}}, color={0,0,255}));
  connect(TH2.p, p2) annotation (Line(points={{20,-10},{20,0},{100,0}}, color={0,0,255}));
  connect(TH1.port_a, heatPort1) annotation (Line(points={{30,60},{30,34},{-60,34},{-60,0},{-100,0}}, color={191,0,0}));
  connect(TH2.port_a, heatPort1)
    annotation (Line(points={{30,-20},{30,34},{-60,34},{-60,0},{-100,0}}, color={191,0,0}));
end ThermistorBridge;
