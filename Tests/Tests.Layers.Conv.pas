unit Tests.Layers.Conv;

interface

uses
  DUnitX.TestFramework,
  Neural.Layers,
  Tests.Layers;

type
  TConvLayerTests = class(TNeuralLayerTest)
  protected
      function BuildLayer(Index: Integer; const ARun: TTestRun): TNeuralLayer; override;
      function GetRuns: System.TArray<Tests.Layers.TTestRun>; override;
  end;

implementation

uses
  Neural,
  Neural.Layers.Conv;

{ TConvLayerTests }

function TConvLayerTests.BuildLayer(Index: Integer;
  const ARun: TTestRun): TNeuralLayer;
begin
  if Index = 0 then
    Result := TConv2D.Create(1, [3, 3], TActivation.Identity, TInitializers.Ones())
  else
    Result := TConv2D.Create(1, [2, 2], TActivation.Identity, TInitializers.Ones())
end;

function TConvLayerTests.GetRuns: System.TArray<Tests.Layers.TTestRun>;
var
  LRun: TTestRun;
  L3DRun: TTestRun;
  LNums: TNums;
begin
  LRun.Input := TNums.Create([4, 4, 1], [
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, -11, -12,
    13, 14, -15, -16
  ]);
  LRun.Output := TNums.Create(
    [2, 2, 1],
    [
      32, 17,
      38, -9
    ]
  );
  LRun.InGradients := TNums.Create(
    [2, 2, 1],
    [
      1, 1,
      1, 1
    ]
  );

  LRun.OutGradients := TNums.Create([4, 4, 1], [
    1, 4, 6, 4,
    10, 24, 28, 16,
    18, 40, -44, -24,
    13, 28, -30, -16
  ]);

  LNums := TNums.Create([2, 2, 2]);
  LNums[0, 0, 0] := 1;
  LNums[1, 0, 0] := 2;
  LNums[0, 1, 0] := 3;
  LNums[1, 1, 0] := 4;
  LNums[0, 0, 1] := 5;
  LNums[1, 0, 1] := 6;
  LNums[0, 1, 1] := 7;
  LNums[1, 1, 1] := 8;
  L3DRun.Input := LNums;
  LNums := TNums.Create([1, 1, 2]);
  LNums[0, 0, 0] := 10;
  LNums[0, 0, 1] := 26;
  L3DRun.Output := LNums;

  LNums := TNums.Create([1, 1, 2]);
  LNums[0, 0, 0] := 1;
  LNums[0, 0, 1] := 1;
  L3DRun.InGradients := LNums;

  LNums := TNums.Create([2, 2, 2]);
  LNums[0, 0, 0] := 1;
  LNums[1, 0, 0] := 2;
  LNums[0, 1, 0] := 3;
  LNums[1, 1, 0] := 4;
  LNums[0, 0, 1] := 5;
  LNums[1, 0, 1] := 6;
  LNums[0, 1, 1] := 7;
  LNums[1, 1, 1] := 8;
  L3DRun.OutGradients := LNums;
  Result := [LRun, L3DRun];
end;

initialization
  TDUnitX.RegisterTestFixture(TConvLayerTests);

end.
