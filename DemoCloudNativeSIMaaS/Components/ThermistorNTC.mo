within DemoCloudNativeSIMaaS.Components;
model ThermistorNTC "NTC thermistor implementing the B parameter equation"
  // https://en.wikipedia.org/wiki/Thermistor#B_or_%CE%B2_parameter_equation
  extends Modelica.Electrical.Analog.Interfaces.OnePort;

  parameter Modelica.SIunits.Temperature T_ref=300.15 "Reference temperature";
  parameter Modelica.SIunits.Resistance R_ref "Resistance at temperature T_ref";
  parameter Modelica.SIunits.Temperature B "B-parameter / 1/K";

  Modelica.SIunits.Resistance R "Temperature-dependant resistance";
  Modelica.SIunits.Temperature T "Temperature of the resistor";
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a port_a
    annotation (Placement(transformation(extent={{-10,-110},{10,-90}})));

equation
  v = R*i;
  1 / T = 1 / T_ref + 1 / B * log(R / R_ref);
  T = port_a.T;
  0 = port_a.Q_flow;

annotation (defaultComponentName="thermistor",
  Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{100,
              100}}), graphics={
          Line(points={{-90,0},{-70,0}}, color={0,0,255}),
          Line(points={{70,0},{90,0}}, color={0,0,255}),
          Rectangle(
            extent={{-70,30},{70,-30}},
            lineColor={0,0,255},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
          Text(
            extent={{-150,90},{150,50}},
            textString="%name",
            lineColor={0,0,255}),
        Line(points={{-70,-40},{-34,-40},{70,42}}, color={28,108,200})}));
end ThermistorNTC;
