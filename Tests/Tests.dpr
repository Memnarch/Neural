program Tests;

{$IFNDEF TESTINSIGHT}
{$APPTYPE CONSOLE}
{$ENDIF}
{$STRONGLINKTYPES ON}
uses
  System.SysUtils,
  {$IFDEF TESTINSIGHT}
  TestInsight.DUnitX,
  {$ELSE}
  DUnitX.Loggers.Console,
  {$ENDIF }
  DUnitX.TestFramework,
  Tests.Layers in 'Tests.Layers.pas',
  Tests.Fixture in 'Tests.Fixture.pas',
  Neural.Layers.Dense in '..\Neural.Layers.Dense.pas',
  Neural.Layers in '..\Neural.Layers.pas',
  Neural.Layers.Pooling in '..\Neural.Layers.Pooling.pas',
  Neural.Math in '..\Neural.Math.pas',
  Neural.Network in '..\Neural.Network.pas',
  Neural in '..\Neural.pas',
  Tests.Layers.Pooling in 'Tests.Layers.Pooling.pas',
  Tests.Layers.Dense in 'Tests.Layers.Dense.pas',
  Neural.Optimizer in '..\Neural.Optimizer.pas',
  Neural.Scheduler in '..\Neural.Scheduler.pas',
  Tests.Layers.Conv in 'Tests.Layers.Conv.pas',
  Neural.Layers.Conv in '..\Neural.Layers.Conv.pas';

var
  runner: ITestRunner;
  results: IRunResults;
  logger: ITestLogger;
  nunitLogger : ITestLogger;
begin
{$IFDEF TESTINSIGHT}
  TestInsight.DUnitX.RunRegisteredTests;
{$ELSE}
  try
    //Check command line options, will exit if invalid
    TDUnitX.CheckCommandLine;
    //Create the test runner
    runner := TDUnitX.CreateRunner;
    //Tell the runner to use RTTI to find Fixtures
    runner.UseRTTI := True;
    //When true, Assertions must be made during tests;
    runner.FailsOnNoAsserts := False;

    //tell the runner how we will log things
    //Log to the console window if desired
    if TDUnitX.Options.ConsoleMode <> TDunitXConsoleMode.Off then
    begin
      logger := TDUnitXConsoleLogger.Create(TDUnitX.Options.ConsoleMode = TDunitXConsoleMode.Quiet);
      runner.AddLogger(logger);
    end;
    //Generate an NUnit compatible XML File
    nunitLogger := TDUnitXXMLNUnitFileLogger.Create(TDUnitX.Options.XMLOutputFile);
    runner.AddLogger(nunitLogger);

    //Run tests
    results := runner.Execute;
    if not results.AllPassed then
      System.ExitCode := EXIT_ERRORS;

    {$IFNDEF CI}
    //We don't want this happening when running under CI.
    if TDUnitX.Options.ExitBehavior = TDUnitXExitBehavior.Pause then
    begin
      System.Write('Done.. press <Enter> key to quit.');
      System.Readln;
    end;
    {$ENDIF}
  except
    on E: Exception do
      System.Writeln(E.ClassName, ': ', E.Message);
  end;
{$ENDIF}
end.
