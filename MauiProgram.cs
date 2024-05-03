using Microsoft.Extensions.Logging;

namespace EmoShift
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

#if DEBUG
    		builder.Logging.AddDebug();
#endif
            // Set current directory to where the app is being run from
            Directory.SetCurrentDirectory(AppDomain.CurrentDomain.BaseDirectory);

#if DEBUG
            Console.WriteLine("Current Dir: " + Directory.GetCurrentDirectory());
#endif

            return builder.Build();
        }
    }
}
