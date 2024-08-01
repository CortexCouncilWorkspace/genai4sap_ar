ALTER VIEW `project_id.dataset_id.AccountingDocumentsReceivable`
ALTER COLUMN Client_MANDT SET OPTIONS (description="Client (Mandant)"),
ALTER COLUMN ExchangeRateType_KURST SET OPTIONS (description="Exchange Rate Type"),
ALTER COLUMN CompanyCode_BUKRS SET OPTIONS (description="Company Code"),
ALTER COLUMN CompanyText_BUTXT SET OPTIONS (description="Company Name"),
ALTER COLUMN CustomerNumber_KUNNR SET OPTIONS (description="Customer"),
ALTER COLUMN FiscalYear_GJAHR SET OPTIONS (description="Fiscal Year"),
ALTER COLUMN NAME1_NAME1 SET OPTIONS (description="Name"),
ALTER COLUMN Company_Country SET OPTIONS (description="Country Key"),
ALTER COLUMN Company_City SET OPTIONS (description="Company"),
ALTER COLUMN CountryKey_LAND1 SET OPTIONS (description="Country Key"),
ALTER COLUMN City_ORT01 SET OPTIONS (description="City"),
ALTER COLUMN AccountingDocumentNumber_BELNR SET OPTIONS (description="Document Number"),
ALTER COLUMN NumberOfLineItemWithinAccountingDocument_BUZEI SET OPTIONS (description="Line item"),
ALTER COLUMN CurrencyKey_WAERS SET OPTIONS (description="Currency"),
ALTER COLUMN LocalCurrency_HWAER SET OPTIONS (description="Local Currency"),
ALTER COLUMN FiscalyearVariant_PERIV SET OPTIONS (description="Fiscal Year Variant"),
ALTER COLUMN Period SET OPTIONS (description="Period"),
ALTER COLUMN Current_Period SET OPTIONS (description="Current"),
ALTER COLUMN AccountType_KOART SET OPTIONS (description="Account Type"),
ALTER COLUMN PostingDateInTheDocument_BUDAT SET OPTIONS (description="Posting Date"),
ALTER COLUMN DocumentDateInDocument_BLDAT SET OPTIONS (description="Document Date"),
ALTER COLUMN InvoiceToWhichTheTransactionBelongs_REBZG SET OPTIONS (description="Invoice reference"),
ALTER COLUMN BillingDocument_VBELN SET OPTIONS (description="Billing Document"),
ALTER COLUMN WrittenOffAmount_DMBTR SET OPTIONS (description="Amt.in loc.cur."),
ALTER COLUMN BadDebt_DMBTR SET OPTIONS (description="Amt.in loc.cur."),
ALTER COLUMN NetDueDate SET OPTIONS (description="Net Due Date"),
ALTER COLUMN CashDiscountDate1 SET OPTIONS (description="Cash Discount Date1"),
ALTER COLUMN CashDiscountDate2 SET OPTIONS (description="Cash Discount Date2"),
ALTER COLUMN OpenAndNotDue SET OPTIONS (description="Open And Not Due"),
ALTER COLUMN ClearedAfterDueDate SET OPTIONS (description="Cleared After Due Date"),
ALTER COLUMN ClearedOnOrBeforeDueDate SET OPTIONS (description="Cleared On Or Before Due Date"),
ALTER COLUMN OpenAndOverDue SET OPTIONS (description="Open And Over Due"),
ALTER COLUMN DoubtfulReceivables SET OPTIONS (description="Doubtful Receivables"),
ALTER COLUMN DaysInArrear SET OPTIONS (description="Days In Arrear"),
ALTER COLUMN AccountsReceivable SET OPTIONS (description="Accounts Receivable"),
ALTER COLUMN Sales SET OPTIONS (description="Sales")