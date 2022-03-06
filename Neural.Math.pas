unit Neural.Math;

interface

function MSELoss(const AExpected, ACalculated: TArray<Single>): Single;
function Sigmoid(const Value: Single): Single;
function DerivedSigmoid(const Value: Single): Single;
function ReLu(const Value: Single): Single;
function DerivedReLu(const Value: Single): Single;

implementation

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

end.
