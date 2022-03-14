unit Tests.Layers.Pooling;

interface

uses
  DUnitX.TestFramework,
  Neural.Layers,
  Tests.Layers;

type
  TMaxPoolingTests = class(TNeuralLayerTest)
  protected
    function BuildLayer(Index: Integer; const ARun: TTestRun): TNeuralLayer; override;
    function GetRuns: TArray<TTestRun>; override;
  end;

implementation

uses
  Windows,
  Neural,
  Neural.Layers.Pooling;

{ TMaxPoolingTests }

function TMaxPoolingTests.BuildLayer(Index: Integer; const ARun: TTestRun): TNeuralLayer;
begin
  Result := TMaxPoolingLayer.Create(Size.Create(2, 2));
end;

function TMaxPoolingTests.GetRuns: TArray<TTestRun>;
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
      6, 8,
      14, -11
    ]
  );
  LRun.InGradients := TNums.Create(
    [2, 2, 1],
    [
      22, 23,
      24, 25
    ]
  );

  LRun.OutGradients := TNums.Create([4, 4, 1], [
    0, 0, 0, 0,
    0, 22, 0, 23,
    0, 0, 25, 0,
    0, 24, 0, 0
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
  LNums[0, 0, 0] := 4;
  LNums[0, 0, 1] := 8;
  L3DRun.Output := LNums;

  LNums := TNums.Create([1, 1, 2]);
  LNums[0, 0, 0] := 22;
  LNums[0, 0, 1] := 23;
  L3DRun.InGradients := LNums;

  LNums := TNums.Create([2, 2, 2]);
  LNums[0, 0, 0] := 0;
  LNums[1, 0, 0] := 0;
  LNums[0, 1, 0] := 0;
  LNums[1, 1, 0] := 22;
  LNums[0, 0, 1] := 0;
  LNums[1, 0, 1] := 0;
  LNums[0, 1, 1] := 0;
  LNums[1, 1, 1] := 23;
  L3DRun.OutGradients := LNums;
  Result := [LRun, L3DRun];
end;


initialization
  TDUnitX.RegisterTestFixture(TMaxPoolingTests);

end.
