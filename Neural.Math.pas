unit Neural.Math;

interface

function MSELoss(const AExpected, ACalculated: TArray<Single>): Single;
function DerivedMSELoss(const AExpected, ACalculated: TArray<Single>): Single;
function Sigmoid(const Value: Single): Single;
function DerivedSigmoid(const Value: Single): Single;
function ReLu(const Value: Single): Single;
function DerivedReLu(const Value: Single): Single;
procedure SoftMax(const Inputs, Outputs: TArray<Single>);

implementation

{$ExcessPrecision OFF}

uses
  System.Math;

function MSELoss(const AExpected, ACalculated: TArray<Single>): Single;
var
  i: Integer;
begin
  Result := 0;
  for i := Low(AExpected) to High(ACalculated) do
    Result := Result + Sqr(AExpected[i] - ACalculated[i]);
  Result := Result / Length(AExpected);
end;

function DerivedMSELoss(const AExpected, ACalculated: TArray<Single>): Single;
var
  i: Integer;
begin
  Result := 0;
  for i := Low(AExpected) to High(ACalculated) do
    Result := Result - 2 * (AExpected[i] - ACalculated[i]);
  Result := Result / Length(AExpected);
end;

function Sigmoid(const Value: Single): Single;
begin
  Result := 1 / (1 + Exp(-Value));
end;

function DerivedSigmoid(const Value: Single): Single;
begin
  Result := Sigmoid(Value);
  Result := Result * (1 - Result);
end;

function ReLu(const Value: Single): Single;
begin
  if Value > 0 then
    Result := Value
  else
    Result := 0;
end;

function DerivedReLu(const Value: Single): Single;
begin
  if Value > 0 then
    Result := 1
  else
    Result := 0;
end;

procedure SoftMax(const Inputs, Outputs: TArray<Single>);
var
  i: Integer;
  LSum: Single;
begin
  LSum := 0;
  for i := Low(Inputs) to High(Inputs) do
  begin
    Outputs[i] := Exp(Inputs[i]);
    LSum := LSum + Outputs[i];
  end;

  for i := Low(Outputs) to High(Outputs) do
    Outputs[i] := Outputs[i] / LSum;
end;

end.
