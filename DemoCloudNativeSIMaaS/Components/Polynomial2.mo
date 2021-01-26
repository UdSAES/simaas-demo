within DemoCloudNativeSIMaaS.Components;
block Polynomial2 "Second order polynomial function"
  extends Modelica.Blocks.Interfaces.SISO;

  parameter Real a "Factor for the quadratic term";
  parameter Real b "Factor for the linear term";
  parameter Real c "Offset";

equation
  y = a*u^2 + b*u + c;
end Polynomial2;
