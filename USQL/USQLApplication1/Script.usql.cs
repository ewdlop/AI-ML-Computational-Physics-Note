using Microsoft.Analytics.Interfaces;
using System.Collections.Generic;
namespace USQL_UDO
{
    public class CountryName : IProcessor
    {
        private static IDictionary<string, string> CountryTranslation = new Dictionary<string, string>
        {
            {
                "Deutschland", "Germany"
            },
            {
                "Suisse", "Switzerland"
            },
            {
                "UK", "United Kingdom"
            },
            {
                "USA", "United States of America"
            },
            {
                "中国", "PR China"
            }
        };
        public override IRow Process(IRow input, IUpdatableRow output)
        {
            string UserID = input.Get<string>("UserID");
            string Name = input.Get<string>("Name");
            string Address = input.Get<string>("Address");
            string City = input.Get<string>("City");
            string State = input.Get<string>("State");
            string PostalCode = input.Get<string>("PostalCode");
            string Country = input.Get<string>("Country");
            string Phone = input.Get<string>("Phone");
            if (CountryTranslation.Keys.Contains(Country))
            {
                Country = CountryTranslation[Country];
            }
            output.Set<string>(0, UserID);
            output.Set<string>(1, Name);
            output.Set<string>(2, Address);
            output.Set<string>(3, City);
            output.Set<string>(4, State);
            output.Set<string>(5, PostalCode);
            output.Set<string>(6, Country);
            output.Set<string>(7, Phone);
            return output.AsReadOnly();
        }
    }
}