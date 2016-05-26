(function(){
	var $ = function(selector){return document.querySelector(selector);};

	if (Enabler.isInitialized()) {
	    dynamicInvocation();
	} else {
	    Enabler.addEventListener(studio.events.StudioEvent.INIT, dynamicInvocation);
	}

	function dynamicInvocation() {
		Enabler.setProfileId(1028052);
	    var devDynamicContent = {};
	    devDynamicContent.offers = [{}];
	    devDynamicContent.offers[0].offerTypeTemplate ="mathstack";//"mathstack"//"aprcashback"//"leasecash";// "lease"//apr"//leasesignndrivecashback
	    devDynamicContent.offers[0].model = "TITAN XD®";//TITAN XD®
	    devDynamicContent.offers[0].modelCode = "XD";
	    devDynamicContent.offers[0].year = "ALL-NEW 2016";
	    devDynamicContent.offers[0].DAB_carImage_728x90 = {};
	    devDynamicContent.offers[0].DAB_carImage_728x90.Url = "https://s0.2mdn.net/ads/richmedia/studio/43353063/9341_20160520150925008_salesEvent_MY16titan_DAB_728x90.png";
	    devDynamicContent.offers[0].background_image_url_728x90 = {};
	    devDynamicContent.offers[0].background_image_url_728x90.Url = "https://s0.2mdn.net/dynamic/2/1256377/motifcdn2.doubleclick.net/Emerging/PPD-Campaign/Nissan/US-Version/Offer/DABimages/Background/background_728x90.jpg_1416952456263_background_728x90.jpg";
	    devDynamicContent.offers[0].baseLandingUrl = {};
	    devDynamicContent.offers[0].baseLandingUrl.Url = "http://www.nissancommercialvehicles.com/financing?tool=global.offers.link";
	    devDynamicContent.offers[0].buildTxt = "BUILD";
	    devDynamicContent.offers[0].baseBuildUrl = {};
	    devDynamicContent.offers[0].baseBuildUrl.Url = "http://www.nissanusa.com/tools/build/altima";
	    devDynamicContent.offers[0].brochureTxt = "SHOP NOW";
	    devDynamicContent.offers[0].baseBrochureUrl = {};
	    devDynamicContent.offers[0].baseBrochureUrl.Url = "http://www.choosenissan.com/altima/";
	    devDynamicContent.offers[0].genDisclaimer = "*More Price Information";
	    devDynamicContent.offers[0].AsShownDisclaimer = "As Shown $42,500 MSRP 2014 Armada 4x2 SL w/splash guards, other optional equipment shown, see dealer for details";
	    devDynamicContent.offers[0].offerRollover_Complete = "*2014 370Z Coupe 370Z Coupe 6M\/T Starting MSRP $29,990. Price excludes tax, title, license, options and destination charge. Dealer sets actual price. See dealer for details.";
	    devDynamicContent.offers[0].offerRollover = "As shown, $43,020 MSRP. 2014 370Z Coupe 370Z Coupe 6M\/T Nismo (other optional equipment shown; see dealer for details).  Excludes tax, title, license, destination charge and options.  Dealer sets actual price. Subject to residency restrictions. <br\/><br\/>2014 370Z Coupe 370Z Coupe 6M\/T Starting MSRP $29,990. Price excludes tax, title, license, options and destination charge. Dealer sets actual price. See dealer for details.  ";

	    devDynamicContent.offers[0].apr_1 = "0.9"//"0.0";
	    devDynamicContent.offers[0].cashbackAPRcopy1 = "APR Financing";

	    devDynamicContent.offers[0].leaseCost1 = "$189";
	    devDynamicContent.offers[0].leaseTerm1 = "PER MONTH LEASE*";
	    devDynamicContent.offers[0].Lease_LeaseCash_OfferCopy = "36 Months, $2,699 Initial Payment, Excl. taxes, title and license";

	    devDynamicContent.offers[0].cashbackAmount = "$1000";
	    devDynamicContent.offers[0].cashbackCopy1 = "President's Day Bonus Cash<sup>[1]</sup>";
	    devDynamicContent.offers[0].cashbackOperatorCopy1 = "PLUS";

	    devDynamicContent.offers[0].offerMsrp1 = "$29,990";
	    devDynamicContent.offers[0].msrpCopy1 = "MSRP*";

	    devDynamicContent.offers[0].offerPrice1 = "Up to $8,000";
	    devDynamicContent.offers[0].offerPrice2 = "$8,000";
	    devDynamicContent.offers[0].offerPrice3 = "";
	    devDynamicContent.offers[0].offerPrice4 = "";
	    devDynamicContent.offers[0].offerCopy1 = "Nissan Cash Back[1]";
	    devDynamicContent.offers[0].offerCopy2 = "NMAC CASH[2]";
	    devDynamicContent.offers[0].offerCopy3 = "Bonus Cash[3]";
	    devDynamicContent.offers[0].offerCopy4 = "";
	    devDynamicContent.offers[0].offerCopy = "As low As";

	    devDynamicContent.offers[0].offernetPrice1 = "$1,500";
	    devDynamicContent.offers[0].offernetPricecopy = "TOTAL SAVING"; //"Total Savings[1]";

	    devDynamicContent.offers[0].leaseTerm2 = "EXCLUDES DEALER FEES, TAXES, TITLE AND LICENSE";
	    devDynamicContent.offers[0].SignnDrive_leaseTerm = "Lease 36 Months, $209 Per Month*";

	    devDynamicContent.offers[0].aprterm1 = "FOR UP TO 60";
		devDynamicContent.offers[0].aprtermcopy2 = "MONTHS";
		devDynamicContent.offers[0].aprtermcopy1 = "FOR WELL-QUALIFIED BUYERS[1]";
		
		//ADDED ON 1/7/2016 for the CY16 Regional Event
		devDynamicContent.offers[0].Sale_carImage_728x90 = {};
		devDynamicContent.offers[0].Sale_carImage_728x90.Url = "https://s0.2mdn.net/ads/richmedia/studio/39674699/9341_20151103152520590_dab_murano_728x90.png";
	    
	    devDynamicContent.salesEvent = [{}];
		
	    devDynamicContent.salesEvent[0].title = "nosale";
	    devDynamicContent.salesEvent[0].saleEventH5_728x90 = {};
	    devDynamicContent.salesEvent[0].saleEventH5_728x90.Url = "https://s0.2mdn.net/ads/richmedia/studio/35158164/9341_20141210074951704_linearGray_728x90.jpg";
	    devDynamicContent.salesEvent[0].swfUrl_728x90 = {};
	    devDynamicContent.salesEvent[0].swfUrl_728x90.Url = "https://s0.2mdn.net/ads/richmedia/studio/40664797/9341_20151228122937048_2016_dab_728x90_e.png";
	    devDynamicContent.salesEvent[0].H5DABIntroVideo_728x90 = "";//"https://s0.2mdn.net/ads/richmedia/studio/appengine/8f9afb2a-7bd4-4750-870a-bd8bcd49a8c9_DABIntroVideo_728x90.mp4|https://s0.2mdn.net/ads/richmedia/studio/appengine/f13fc4b8-ffda-4ad5-9069-a8c66fd94d6a_DABIntroVideo_728x90.ogv|https://s0.2mdn.net/ads/richmedia/studio/appengine/45a65bd1-57c5-4442-aa8d-7e84ee5fe4c1_DABIntroVideo_728x90.webm";
	    devDynamicContent.salesEvent[0].textColor = "";

	    devDynamicContent.salesEvent[0].model_year_prefix_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].nissan_copy_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].apr_up_to_copy = "#2E64FE";
	    devDynamicContent.salesEvent[0].cashbackOperator_static_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].leasesigndrive_0_static_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].leasesigndrive_text_static_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].line_divider_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].model_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].year_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].apr_1_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].aprtermcopy1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].aprtermcopy2_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].aprterm1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].cashbackAPRcopy1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].cashbackOperatorCopy1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].cashbackCopy1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].cashbackAmount_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].leaseCost1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].leaseTerm1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].leaseTerm2_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].SignnDrive_leaseTerm_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerCopy_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].Lease_LeaseCash_OfferCopy_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerCopy1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerCopy2_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerCopy3_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerCopy4_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offernetPricecopy_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerMsrp1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].msrpCopy1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offernetPrice1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerPrice1_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerPrice2_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerPrice3_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerPrice4_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].AsShownDisclaimer_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].genDisclaimer_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].offerRollover_Complete_color_hander = "#2E64FE";
	    devDynamicContent.salesEvent[0].buildTxt_color_handler = "#2E64FE";
	    devDynamicContent.salesEvent[0].brochureTxt_color_handler = "#2E64FE";
		
		devDynamicContent.salesEvent[0].SpecialSale_carImage_728x90 = {};
		devDynamicContent.salesEvent[0].SpecialSale_carImage_728x90.Url = "https://s0.2mdn.net/ads/richmedia/studio/37028747/9341_20150612122445167_altima_728x90_template.jpg";

	    //MODEL CODE and OFFERTYPE FILTER COLUMNS
		devDynamicContent.salesEvent[0].modelCodeFilter = "SE,MU,FT,ZC,AS";
		devDynamicContent.salesEvent[0].offerTypeFilter = "lease,leasecash,msrp";
		
	    Enabler.setDevDynamicContent(devDynamicContent);

		var carExitURL1 = [];
		var carExitURL2 = [];
		carExitURL1 = dynamicContent.offers[0].baseBrochureUrl.Url.split("/");
		carExitURL2 = dynamicContent.offers[0].baseBuildUrl.Url.split("/");
		brochureExit = carExitURL1[0]+ "//" + carExitURL1[2] + "/";
		baseLandingUrl = carExitURL2[0]+ "//" + carExitURL2[2] + "/";
		
        //Enabler.loadScript("https://s0.2mdn.net/ads/richmedia/studio/36493392/9341_20150428192342420_dab_dmaRegional.js");
        Enabler.loadScript("https://s0.2mdn.net/ads/richmedia/studio/42214003/dab_dmaRegional.js");
	    
		return new Offer();
	};

	function Offer(){
		console.log(this);

		//ADD MODEL AND OFFERTYPE FILTERS TO THE SALEEVENT FEED
		//ADDED ON 1/11/2016
		var offerType = dynamicContent.salesEvent[0].offerTypeFilter.split(",");
		var model = dynamicContent.salesEvent[0].modelCodeFilter.split(",");

		if( ( offerType.length > -1 || model.length > -1) && (  offerType.indexOf( dynamicContent.offers[0].offerTypeTemplate ) > -1 || model.indexOf( dynamicContent.offers[0].modelCode ) > -1 ) ) {
			dynamicContent.salesEvent[0].saleEventH5_728x90.Url = dynamicContent.salesEvent[0].swfUrl_728x90.Url;
			dynamicContent.offers[0].Sale_carImage_728x90.Url = dynamicContent.offers[0].Sale_carImage_728x90.Url;
		} else {
			dynamicContent.salesEvent[0].saleEventH5_728x90.Url = dynamicContent.salesEvent[0].saleEventH5_728x90.Url;
			dynamicContent.offers[0].Sale_carImage_728x90.Url =  dynamicContent.offers[0].DAB_carImage_728x90.Url;
		}

		//END OF ADDED LINES

		//dynamicContent.salesEvent[0].saleEventH5_728x90.Url = (dynamicContent.offers[0].modelCode == "AS" && ( dynamicContent.salesEvent[0].title == "royl" || dynamicContent.salesEvent[0].title == "cy16" ) ) ? dynamicContent.salesEvent[0].swfUrl_728x90.Url : dynamicContent.salesEvent[0].saleEventH5_728x90.Url;
		dynamicContent.offers[0].background_image_url_728x90.Url = ( dynamicContent.offers[0].modelCode == "AS" ) ? dynamicContent.salesEvent[0].SpecialSale_carImage_728x90.Url : dynamicContent.offers[0].background_image_url_728x90.Url;
		this.content = "";
    	this.offer = dynamicContent.offers[0].offerTypeTemplate;
    	this.model = dynamicContent.offers[0].model.split("®").join("<sup>\u00AE</sup>");
    	this.year = dynamicContent.offers[0].year;
    	this.salesEvent = dynamicContent.salesEvent[0].title;
        this.salesEventImg = (this.salesEvent == "l4l" || this.salesEvent == "minievent" || this.salesEvent == "nosale") ? dynamicContent.offers[0].background_image_url_728x90.Url : dynamicContent.salesEvent[0].saleEventH5_728x90.Url;
        this.bgImg = (this.salesEventImg.length > 0 && this.salesEventImg.indexOf("Filler") < 0 && this.salesEventImg.indexOf("filler") < 0) ? "background-image:url("+ this.salesEventImg +")":"background-image:url("+ dynamicContent.offer[0].background_image_url_728x90.Url +")";
		
		//ADDED on 1/7/2016
		//GET THE CAR IMAGE LINK OF THE DAB_carImage_728x90 COLUMN IF TRUE, IF FALSE, GET THE Sale_carImage_728x90 LINK IN THE FEED
		this.salesCarImg = (this.salesEvent == "l4l" || this.salesEvent == "minievent" || this.salesEvent == "nosale") ? dynamicContent.offers[0].DAB_carImage_728x90.Url : dynamicContent.offers[0].Sale_carImage_728x90.Url;
		this.carImg = ( this.salesCarImg.length > 0 && this.salesCarImg.indexOf( "Filler" ) < 0 && this.salesCarImg.indexOf( "filler" ) < 0 ) ? "background-image:url("+ this.salesCarImg +")" : "background-image:url("+ dynamicContent.offers[0].DAB_carImage_728x90.Url +")"
		
		
    	// this.carImg = (dynamicContent.offers[0].DAB_carImage_728x90.Url.length > 0)? "background-image:url(" + dynamicContent.offers[0].DAB_carImage_728x90.Url + ")":"background-color:rgba(0,0,0,0)";
        this.landingUrl = dynamicContent.offers[0].baseLandingUrl.Url;
    	this.buildUrl = dynamicContent.offers[0].baseBuildUrl.Url;
    	this.buildTxt = dynamicContent.offers[0].buildTxt;
    	this.brochureTxt = dynamicContent.offers[0].brochureTxt;
    	this.brochureUrl = dynamicContent.offers[0].baseBrochureUrl.Url;
    	this.disclaimerTxt = (dynamicContent.offers[0].genDisclaimer.indexOf("*") >= 0) ? dynamicContent.offers[0].genDisclaimer : "*" + dynamicContent.offers[0].genDisclaimer;
    	this.asShownDisclaimerTxt = dynamicContent.offers[0].AsShownDisclaimer;
		this.offerRollover = "*" + dynamicContent.offers[0].offerRollover;
	
    	disclaimerContentL4L = "This information does not constitute tax or legal advice.  Lease price depends on individual circumstances, including the availability of certain tax credits.  All persons should consult with their own tax and/or legal professional and state law to determine eligibility and any further details.<br>";
    	this.disclaimerContent = (this.salesEvent == "l4l")?(this.offer == "lease" || this.offer == "leasecash") ? disclaimerContentL4L + this.offerRollover : this.offerRollover : this.offerRollover;
 
 		//General Colors
        this.model_year_prefix_color = dynamicContent.salesEvent[0].model_year_prefix_color_handler;
        this.nissan_copy_color = dynamicContent.salesEvent[0].nissan_copy_color_handler;
        this.apr_up_to_copy = dynamicContent.salesEvent[0].apr_up_to_copy;
        this.line_divider_color = dynamicContent.salesEvent[0].line_divider_color_handler;
        this.model_color = dynamicContent.salesEvent[0].model_color_handler;
        this.year_color = dynamicContent.salesEvent[0].year_color_handler;
        this.genDisclaimer_color = dynamicContent.salesEvent[0].genDisclaimer_color_hander;
        this.AsShownDisclaimer_color = dynamicContent.salesEvent[0].AsShownDisclaimer_color_hander;
        this.offerRollover_Complete_color = dynamicContent.salesEvent[0].offerRollover_Complete_color_hander;
        this.cashbackOperator_static_color = dynamicContent.salesEvent[0].cashbackOperator_static_color_handler;
    	//apr
		this.apr = (dynamicContent.offers[0].apr_1 == 0) ? "0" : dynamicContent.offers[0].apr_1;
		this.aprCopy = dynamicContent.offers[0].cashbackAPRcopy1;
		// this.aprTerm = dynamicContent.offers[0].aprterm1 + " " + dynamicContent.offers[0].aprtermcopy2 + "<BR>" + dynamicContent.offers[0].aprtermcopy1;
		this.aprTerm = dynamicContent.offers[0].aprterm1 + " " + dynamicContent.offers[0].aprtermcopy2;
		this.aprTerm2 = dynamicContent.offers[0].aprtermcopy1;
		this.apr_1_color = dynamicContent.salesEvent[0].apr_1_color_handler;
        this.cashbackAPRcopy1_color = dynamicContent.salesEvent[0].cashbackAPRcopy1_color_hander;
        this.aprtermcopy1_color = dynamicContent.salesEvent[0].aprtermcopy1_color_hander;
        this.aprtermcopy2_color = dynamicContent.salesEvent[0].aprtermcopy2_color_hander; // same as aprterm1_color_hander
		//lease
		this.leaseCost = dynamicContent.offers[0].leaseCost1.split("$").join("<sup>$</sup>");
		this.leaseTerm = dynamicContent.offers[0].leaseTerm1;
		this.leaseCopy = dynamicContent.offers[0].Lease_LeaseCash_OfferCopy;
		this.leaseCost1_color = dynamicContent.salesEvent[0].leaseCost1_color_hander;
        this.leaseTerm1_color = dynamicContent.salesEvent[0].leaseTerm1_color_hander;
        this.Lease_LeaseCash_OfferCopy_color = dynamicContent.salesEvent[0].Lease_LeaseCash_OfferCopy_color_hander;
		//leasecash
		this.cashbackAmount = (this.offer == "leasesignndrivecashback") ? dynamicContent.offers[0].cashbackAmount : dynamicContent.offers[0].cashbackAmount.split("$").join("<sup>$</sup>");
		this.cashbackCopy = dynamicContent.offers[0].cashbackCopy1;
		this.cashbackAmount_color = dynamicContent.salesEvent[0].cashbackAmount_color_hander;
        this.cashbackCopy1_color = dynamicContent.salesEvent[0].cashbackCopy1_color_hander;
		//cashback
		this.netPrice = dynamicContent.offers[0].offernetPrice1;
		this.netCopy = dynamicContent.offers[0].offernetPricecopy;
		this.operatorCopy = dynamicContent.offers[0].cashbackOperatorCopy1;
		this.cashbackOperatorCopy1_color = dynamicContent.salesEvent[0].cashbackOperatorCopy1_color_hander;
        this.offernetPrice1_color = dynamicContent.salesEvent[0].offernetPrice1_color_hander;
        this.offernetPricecopy_color = dynamicContent.salesEvent[0].offernetPricecopy_color_hander;
		//msrp
		this.msrp = dynamicContent.offers[0].offerMsrp1.split("$").join("<sup>$</sup>");
		this.msrpCopy = dynamicContent.offers[0].msrpCopy1;
		this.offerCopy_color = dynamicContent.salesEvent[0].offerCopy_color_hander;
        this.offerMsrp1_color = dynamicContent.salesEvent[0].offerMsrp1_color_hander;
        this.msrpCopy1_color = dynamicContent.salesEvent[0].msrpCopy1_color_hander;
		//mathstack
		this.posOfferPrice = [dynamicContent.offers[0].offerPrice1.toLowerCase().replace("up to",""),
							dynamicContent.offers[0].offerPrice2,
							dynamicContent.offers[0].offerPrice3,
							dynamicContent.offers[0].offerPrice4];

		this.posOfferPriceColor = [dynamicContent.salesEvent[0].offerPrice1_color_hander,
                            dynamicContent.salesEvent[0].offerPrice2_color_hander,
                            dynamicContent.salesEvent[0].offerPrice3_color_hander,
                            dynamicContent.salesEvent[0].offerPrice4_color_hander];

		this.posOfferCopy = [dynamicContent.offers[0].offerCopy1,
					dynamicContent.offers[0].offerCopy2,
					dynamicContent.offers[0].offerCopy3,
					dynamicContent.offers[0].offerCopy4];

		this.posOfferCopyColor = [dynamicContent.salesEvent[0].offerCopy1_color_hander,
                    dynamicContent.salesEvent[0].offerCopy2_color_hander,
                    dynamicContent.salesEvent[0].offerCopy3_color_hander,
                    dynamicContent.salesEvent[0].offerCopy4_color_hander];

		this.negOfferPrice = [dynamicContent.offers[0].offerMsrp1,
							dynamicContent.offers[0].offerPrice1.toLowerCase().replace("up to",""),
							dynamicContent.offers[0].offerPrice2,
							dynamicContent.offers[0].offerPrice3];

		this.negOfferPriceColor = [dynamicContent.salesEvent[0].offerMsrp1_color_hander,
                            dynamicContent.salesEvent[0].offerPrice1_color_hander,
                            dynamicContent.salesEvent[0].offerPrice2_color_hander,
                            dynamicContent.salesEvent[0].offerPrice3_color_hander];

		this.negOfferCopy = [dynamicContent.offers[0].msrpCopy1,
					dynamicContent.offers[0].offerCopy1,
					dynamicContent.offers[0].offerCopy2,
					dynamicContent.offers[0].offerCopy3];

		this.negOfferCopyColor = [dynamicContent.salesEvent[0].msrpCopy1_color_hander,
                    dynamicContent.salesEvent[0].offerCopy1_color_hander,
                    dynamicContent.salesEvent[0].offerCopy2_color_hander,
                    dynamicContent.salesEvent[0].offerCopy3_color_hander];

		this.offerPrice = (this.offer == "mathstack") ? this.posOfferPrice : (this.offer == "negmathstack") ? this.negOfferPrice : "";
		this.offerCopy = (this.offer == "mathstack") ? this.posOfferCopy: (this.offer == "negmathstack") ? this.negOfferCopy : "";
		this.offerPriceColor = (this.offer == "mathstack") ? this.posOfferPriceColor : (this.offer == "negmathstack") ? this.negOfferPriceColor : "";
        this.offerCopyColor = (this.offer == "mathstack") ? this.posOfferCopyColor: (this.offer == "negmathstack") ? this.negOfferCopyColor : "";

		this.headCopy = dynamicContent.offers[0].offerCopy;
		//snd
		this.leaseTerm_snd = dynamicContent.offers[0].leaseTerm2;
		this.snd_leaseTerm = dynamicContent.offers[0].SignnDrive_leaseTerm;
		this.leasesigndrive_text_static_color = dynamicContent.salesEvent[0].leasesigndrive_text_static_color_handler;
        this.leasesigndrive_0_static_color = dynamicContent.salesEvent[0].leasesigndrive_0_static_color_handler;
        this.leaseTerm2_color = dynamicContent.salesEvent[0].leaseTerm2_color_hander;
        this.SignnDrive_leaseTerm_color = dynamicContent.salesEvent[0].SignnDrive_leaseTerm_color_hander;
		//
		//this.videoURLS = dynamicContent.salesEvent[0].H5DABIntroVideo_728x90;

	    this.createContent();
		this.addClick();
	}

	Offer.prototype.createContent = function() {
		$('.container').classList.add(this.offer);
		$('.container').setAttribute('style', this.bgImg);
		this.content += "<div class='carImg' style=" + this.carImg + ">";
		this.content += "<div class='logo'></div>";//logo
		this.addOfferContent();
		this.content += "<div class='model'>"
		this.content += (this.model.toUpperCase().indexOf("LEAF") >= 0 || dynamicContent.offers[0].modelCode == "XD") ? "<div class='year' style='font-size:9px!important'><span style=color:"+this.model_year_prefix_color+">THE</span> <span style=color:"+this.year_color+">" + this.year + "</span></div>" : "<div class='year'><span style=color:"+this.model_year_prefix_color+">THE</span> <span style=color:"+this.year_color+">" + this.year + "</span> <a style='color:rgb(51, 51, 51);'><span style=color:"+this.nissan_copy_color+">NISSAN</span></a></div>";
		this.content += (this.model.toUpperCase().indexOf("LEAF") >= 0 || dynamicContent.offers[0].modelCode == "XD") ? "<div class='name'><span style=color:"+this.nissan_copy_color+">NISSAN</span> <span style=color:"+this.model_color+">" + this.model + "</span></div>" : "<div class='name'><span style=color:"+this.model_color+">" + this.model + "</span></div>";
		this.content += "</div>"//model year and name
		this.content += "<div class='cta cta1'>";
		this.content +=	"<div class='ctaTxt'>" + this.brochureTxt + "</div>";
		this.content +=	"</div>";//brochure button
		this.content += "<div class='cta cta2'>";
		this.content +=	"<div class='ctaTxt'>" + this.buildTxt + "</div>";
		this.content +=	"</div>";//build button
		this.content +=	"<div class='disclaimerTxt' style=color:"+this.genDisclaimer_color+">" + this.disclaimerTxt + "</div>";//dialog text
		this.content += (this.offer == "lease" || this.offer == "leasecash" || this.offer == "leasecash" || this.offer == "msrp" || this.offer == "mathstack" || this.offer == "negmathstack") ? "<div class='asShownDisclaimerTxt'><span style=color:"+this.AsShownDisclaimer_color+">" + this.asShownDisclaimerTxt + "</span></div>":"";
		this.content +=	"<div class='disclaimerBox'>";
		this.content +=	"<div class='closeBtn'></div>";//close button
		this.content +=	"<div class='disclaimerContent' style=color:"+this.offerRollover_Complete_color+">" + this.disclaimerContent + "</div>";//dialog content
		this.content +=	"</div>";//dialog box
		this.content += "</div>";
		//this.addVideo();
		$(".container").innerHTML = this.content;

		this.setContent();
	};

	//Offer.prototype.addVideo = function() {
	//	if(this.videoURLS.length > 0 && this.videoURLS.indexOf("Filler") < 0 && this.videoURLS.indexOf("filler") < 0){
	//		var videoURL = this.videoURLS.split("|");
	//		var extension;
	//		console.log(videoURL);
	//
	//		this.content += "<div class='videobox'>"
	//		this.content += "<video class='vid' autoplay>";
	//		for (var i = 0; i < videoURL.length; i++) {
	//			extension = videoURL[i].substr(videoURL[i].lastIndexOf(".")+1, videoURL[i].length);
	//			this.content += "<source src=" + videoURL[i] + " type='video/"+ extension +"'>";
	//		}
	//		this.content += "</video>";
	//		this.content += "</div>"
	//	}
	//
	//	return this.content;
	//};
	//
	//Offer.prototype.setVideo = function() {
	//	if(this.videoURLS.length > 0 && this.videoURLS.indexOf("Filler") < 0 && this.videoURLS.indexOf("filler") < 0){
	//        Enabler.loadModule(studio.module.ModuleId.VIDEO, function() {
	//            studio.video.Reporter.attach('video_play', $(".vid"));
	//        });
	//
	//        $(".vid").addEventListener('ended',function(){
	//        	this.style.display = "none";
	//        	$(".videobox").addEventListener('transitionend',function(){
	//        		this.style.display = "none";
	//        	});
	//        	$(".videobox").classList.add('fadeOut');
	//        });
	//    }
	//};

	Offer.prototype.priceCount = function() {
		var priceNum = 0;
		for(var i in this.offerPrice){
			this.offerPrice[i] = this.offerPrice[i].replace(/^\s+|\s+$/gm,'');
			(this.offerPrice[i] || '')? priceNum++:priceNum;
		}
		return priceNum;
	};

	Offer.prototype.addOfferContent = function() {
		switch(this.offer){
			case 'apr':
				this.content += "<div class='content'>"
					this.content += "<div class='table'>"
						this.content += "<div class='tr'>"
							this.content += "<div class='rate td' style=color:"+this.apr_1_color+">" + this.apr + "</div>";
							this.content += "<div class='sup td'>";
								this.content += "<div style=color:"+this.apr_1_color+">%</div>";
								this.content += "<div style=color:"+this.cashbackAPRcopy1_color+">" + this.aprCopy;
								this.content += "<span style=color:"+this.aprtermcopy2_color+">" + this.aprTerm + "</span>"
								this.content += "</div>";
								this.content += "<div style=color:"+this.aprtermcopy1_color+">" + this.superScriptBracket(this.aprTerm2) + "</div>";//term
							this.content += "</div>";
						this.content += "</div>";
					this.content += "</div>";
				this.content += "</div>";

			break;
			case 'cashback':
				this.content += "<div class='content'>";
					this.content += "<div class='table'>"
						this.content += (this.netCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0) ? "<p class='amountUpto' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</p>" : "<p class='amount' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</p>";
						this.content += "<p class='copy'><span style=color:"+this.offernetPricecopy_color+">" + this.superScriptBracket(this.netCopy) + "</span></p>";
					this.content += "</div>";
				this.content += "</div>";
			break;
			case 'lease':
				this.content += "<div class='content'>"
					this.content += "<div class='table'>"
						this.content += "<div class='tr'>"
							this.content += "<div class='amount td' style=color:"+this.leaseCost1_color+">" + this.leaseCost + "</div>";
							this.content += "<div class='copy td'>";
								this.content += "<p style=color:"+this.leaseTerm1_color+">" + this.leaseTerm + "</p>";
								this.content += "<p style=color:"+this.Lease_LeaseCash_OfferCopy_color+">" + this.leaseCopy + "</p>";
							this.content += "</div>";
						this.content += "</div>";
					this.content += "</div>";
				this.content += "</div>";
			break;
			case 'leasecash':
				this.content += "<div class='content'>"
					this.content += "<div class='table'>";
						this.content += "<div class='tr'>";

							this.content += "<div class='leaseContent td' style=border-color:"+this.line_divider_color+">"
								this.content += "<div class='leaseWrapper'>"
									this.content += "<p class='amount' style=color:"+this.leaseCost1_color+">" + this.leaseCost + "</p>";
									this.content += "<p style=color:"+this.leaseTerm1_color+">" + this.leaseTerm + "</p>";
									this.content += "<p style=color:"+this.Lease_LeaseCash_OfferCopy_color+">" + this.leaseCopy + "</p>";
								this.content += "</div>";
							this.content += "</div>";

							this.content += "<div class='cashbackContent td'>";
								this.content += "<div class='cashbackWrapper'>"
									this.content += "<p class='operatorCopy' style=color:"+this.cashbackOperator_static_color+">PLUS</p>";
									this.content += (this.cashbackCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0) ? "<p class='upTo' style=color:"+this.apr_up_to_copy+">Up To</p>" : "";
									this.content += "<p class='amount' style=color:"+this.cashbackAmount_color+">" + this.cashbackAmount + "</p>";
									this.content += "<p class='copy' style=color:"+this.cashbackCopy1_color+">" + this.superScriptBracket(this.cashbackCopy) + "</p>";
								this.content += "</div>";
							this.content += "</div>";

						this.content += "</div>";
					this.content += "</div>";
				this.content += "</div>";
			break;
			case 'aprcashback':
				this.content += "<div class='content'>"
					this.content += "<div class='table'>";
						this.content += "<div class='tr'>";

							this.content += "<div class='aprContent td' style=border-color:"+this.line_divider_color+">"
								this.content += "<div class='aprWrapper'>"
									this.content += "<div class='rate' style=color:"+this.apr_1_color+">" + this.apr;
									this.content += "<sup style=color:"+this.apr_1_color+">%</sup>"
									this.content += "</div>";
									this.content += "<p style=color:"+this.cashbackAPRcopy1_color+">" + this.aprCopy;//static : APR FINANCING
										this.content += "<span style=color:"+this.aprtermcopy2_color+">" + this.aprTerm + "</span>";
									this.content += "</p>";//static : APR FINANCING
									this.content += "<p style=color:"+this.aprtermcopy1_color+">" + this.superScriptBracket(this.aprTerm2) + "</p>";
								this.content += "</div>";
							this.content += "</div>";

							this.content += "<div class='cashbackContent td'>";
								this.content += "<div class='cashbackWrapper'>"
									this.content += "<p class='operatorCopy' style=color:"+this.cashbackOperatorCopy1_color+">" + this.operatorCopy + "</p>";
									this.content += (this.cashbackCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0) ? "<p class='upTo' style=color:"+this.apr_up_to_copy+">Up To</p>" : "";
									this.content += "<p class='amount' style=color:"+this.cashbackAmount_color+">" + this.cashbackAmount + "</p>";
									this.content += "<p class='copy' style=color:"+this.cashbackCopy1_color+">" + this.superScriptBracket(this.cashbackCopy) + "</p>";
								this.content += "</div>";
							this.content += "</div>";

						this.content += "</div>";
					this.content += "</div>";
				this.content += "</div>";
			break;
			case 'msrp':
				this.content += "<div class='content'>"
					this.content += "<div class='table'>"
						this.content += "<div class='amount' style=color:"+this.offerMsrp1_color+">" + this.msrp;
						this.content += "<div class='copy' style=color:"+this.offerCopy_color+">" + this.headCopy+ "</div>";
						this.content += "</div>";
						this.content += "<div class='after' style=color:"+this.msrpCopy1_color+">" + this.msrpCopy + "</div>";
					this.content += "</div>";
				this.content += "</div>";
			break;
			case 'mathstack':
				this.content += "<div class='content'>";
					this.content += "<div class='table'>";
					if(this.priceCount() > 1 && this.priceCount() < 4){
						this.content += "<div class='equationWrapper'>";
						for (var i = 0; i < this.priceCount(); i++) {
							this.content += "<div class='amount' style=color:"+this.offerPriceColor[i]+">" + this.checkCurrency(this.offerPrice[i]) + "</div>";
							this.content += "<div class='copy' style=color:"+this.offerCopyColor[i]+"><span>" + this.superScriptBracket(this.offerCopy[i]) + "</span></div>";
							this.content +=	(i < this.priceCount()-1)?"<span class='ope' style=color:"+this.offerPriceColor[i]+">+</span>":"";
						}
						this.content += "</div>";
						this.content += "<div class='totalWrapper'>"; 
							this.content += (this.netCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0) ? "<div class='upTo' style=color:"+this.apr_up_to_copy+">Up To</div>" : (this.headCopy.toUpperCase().indexOf("UP TO") >= 0) ? "" : "<div class='upTo' style=color:"+this.offerCopy_color+">" + this.headCopy + "</div>";
							this.content += "<div class='net' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</div>";
							this.content += "<div class='copy' style=color:"+this.offernetPricecopy_color+">" + this.superScriptBracket(this.netCopy) + "</div>";
						this.content += "</div>";
					}else{
						this.content += (this.netCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0) ? "<p class='amountCondUpto' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</p>" : "<p class='amountCond' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</p>";
						this.content += "<div class='copyCond'><span style=color:"+this.offernetPricecopy_color+">" + this.superScriptBracket(this.netCopy) + "</span></div>";
					}
					this.content += "</div>";
				this.content += "</div>";
			break;
			case 'negmathstack':
				this.content += "<div class='content'>";
					this.content += "<div class='table'>";
					if(this.priceCount() > 1 && this.priceCount() < 4){
						this.content += "<div class='equationWrapper'>";
						for (var i = 0; i < this.priceCount(); i++) {
							this.content += "<div class='amount' style=color:"+this.offerPriceColor[i]+">" + this.checkCurrency(this.offerPrice[i]) + "</div>";
							this.content += "<div class='copy'><span style=color:"+this.offerCopyColor[i]+">" + this.superScriptBracket(this.offerCopy[i]) + "</span></div>";
							this.content +=	(i < this.priceCount()-1)?"<span class='ope' style=color:"+this.offerPriceColor[i]+">-</span>":"";
						}
						this.content += "</div>";
						this.content += "<div class='totalWrapper'>";
							this.content += (this.netCopy.toUpperCase().indexOf("TOTAL SAVINGS") >= 0) ? "<div class='upTo' style=color:"+this.apr_up_to_copy+">Up To</div>" : (this.headCopy.toUpperCase().indexOf("UP TO") >= 0) ? "" : "<div class='upTo' style=color:"+this.offerCopy_color+">" + this.headCopy + "</div>";
							this.content += "<div class='net' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</div>";
							this.content += "<div class='copy' style=color:"+this.offernetPricecopy_color+">" + this.superScriptBracket(this.netCopy) + "</div>";
						this.content += "</div>";
					} else {
						this.content += (this.netCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0) ? "<p class='amountCondUpto' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</p>" : "<p class='amountCondAs' style=color:"+this.offernetPrice1_color+">" + this.checkCurrency(this.netPrice) + "</p>";
						this.content += "<div class='copyCond'><span style=color:"+this.offernetPricecopy_color+">" + this.superScriptBracket(this.netCopy) + "</span></div>";
					}
					this.content += "</div>";
				this.content += "</div>";
			break;
			case 'leasesignndrive':
				this.content += "<div class='table'>";
				this.content += "<div class='tr'>";
				this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
				this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Initial Payment<sup>[1]</sup></span></div>";
				this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
				this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Down Payment<sup>[2]</sup></span></div>";
				this.content += "</div>";
				this.content += "<div class='tr'>";
				this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
				this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Security Deposit<sup>[3]</sup></span></div>";
				this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
				this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Drive-off Lease<sup>[4]</sup></span></div>";
				this.content += "</div>";
				this.content += "</div>";//add table
				this.content += "<p class='copy' style=color:"+this.leaseTerm2_color+">" + this.snd_leaseTerm.toUpperCase() + "</p>";
				this.content += "<p style=color:"+this.SignnDrive_leaseTerm_color+">" + this.leaseTerm_snd + "</p>";
			break;
		    case 'leasesignndrivecashback':
			    this.content += "<div class='table'>";
			    this.content += "<div class='tr'>";
			    this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
			    this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Initial Payment<sup>[1]</sup></span></div>";
			    this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
		     	this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Down Payment<sup>[2]</sup></span></div>";
		    	this.content += "</div>";
		    	this.content += "<div class='tr'>";
		    	this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
		    	this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Security Deposit<sup>[3]</sup></span></div>";
		    	this.content += "<div class='td amount' style=color:"+this.leasesigndrive_0_static_color+"><sup>$</sup>0</div>";
		    	this.content += "<div class='td copy' style=color:"+this.leasesigndrive_text_static_color+"><span>Drive-off Lease<sup>[4]</sup></span></div>";
		    	this.content += "</div>";
		    	this.content += "</div>";//add table
				this.content += "<p class='bonuscopy' style=color:"+ this.SignnDrive_leaseTerm_color +">+ "+ this.cashbackAmount +" "+" "+ this.cashbackCopy.toUpperCase() + "</p>";//dynamic content within the <p> tag should be "this.cashbackAmount + this.cashbackCopy"
		    	this.content += "<p class='copy' style=color:"+this.leaseTerm2_color+">" + this.snd_leaseTerm.toUpperCase() + "</p>";
		    	this.content += "<p style=color:"+this.SignnDrive_leaseTerm_color+">" + this.leaseTerm_snd + "</p>";
		    break;
		}
	};

	Offer.prototype.setContent = function() {
		switch(this.offer){
			case 'apr':
				this.centerValignText(".table");
				this.alignCenter(".table",".content");
				$('.sup div:nth-child(2)').style.marginTop = ($('.sup div:nth-child(2)').offsetHeight > 15 || $('.sup div:nth-child(3)').offsetHeight > 11) ? "0px" : "8px";
			break;
			case 'cashback':
				$("p.copy").style.width = ($("p.copy").offsetWidth > $("p.copy span").offsetWidth) ? $("p.copy span").offsetWidth + 1 + "px" : $("p.copy").offsetWidth;
				$(".content").style.marginLeft = (this.netCopy.toUpperCase().indexOf("TOTAL SAVINGS") >= 0) ? "15px" : "0";
				this.alignCenter(".table",".content");
				this.centerValignText(".table");
			break;
			case 'lease':
				this.alignCenter(".table",".content");
				this.centerValignText(".table");
				$(".content").style.marginTop = (this.asShownDisclaimerTxt.length > 0) ? "-5px" : "0px";
				$(".asShownDisclaimerTxt").style.left = $(".content").offsetLeft + $(".table").offsetLeft + "px";
				$(".asShownDisclaimerTxt").style.width = $(".table").offsetWidth + (($(".content").offsetWidth - $(".table").offsetWidth) / 2) - 5 + "px";
				$(".asShownDisclaimerTxt").style.top = $(".content").offsetHeight - $(".table").offsetTop + $(".content").offsetTop + "px";
				$(".asShownDisclaimerTxt").style.bottom = "0px";
				this.centerValignText(".asShownDisclaimerTxt span");
			break;
			case 'leasecash':
				this.alignCenter(".table",".content");
				this.centerValignText(".table");
				$(".leaseWrapper").style.top = ($(".table").offsetHeight - ($(".leaseWrapper").offsetHeight - 11)) / 2 + "px";
				$(".cashbackWrapper").style.top = ($(".table").offsetHeight - $(".cashbackWrapper").offsetHeight) / 2 + "px";
				$(".content").style.top = (this.asShownDisclaimerTxt.length > 0) ? "-8px" : "0px";
				$(".asShownDisclaimerTxt").style.left = $(".content").offsetLeft + $(".table").offsetLeft + "px";
				$(".asShownDisclaimerTxt").style.top = $(".content").offsetHeight - $(".table").offsetTop + $(".content").offsetTop + "px";
				$(".asShownDisclaimerTxt").style.bottom = "0px";
				$(".asShownDisclaimerTxt").style.width = $(".table").offsetWidth + (($(".content").offsetWidth - $(".table").offsetWidth) / 2) - 5 + "px";
				this.centerValignText(".asShownDisclaimerTxt span");
			break;
			case 'aprcashback':
				this.alignCenter(".table",".content");
				this.centerValignText(".table");
				$(".aprWrapper").style.top = ($(".table").offsetHeight - ($(".aprWrapper").offsetHeight - 11)) / 2 + "px";
				$(".cashbackWrapper").style.top = ($(".table").offsetHeight - $(".cashbackWrapper").offsetHeight) / 2 + "px";
			break;
			case 'msrp':
				this.alignCenter(".table",".content");
				this.centerValignText(".table");
				$(".content").style.marginTop = (this.asShownDisclaimerTxt.length > 0) ? "-3px" : "0px";
				$(".asShownDisclaimerTxt").style.left = $(".content").offsetLeft + $(".table").offsetLeft + "px";
				$(".asShownDisclaimerTxt").style.right = "150px";
				$(".asShownDisclaimerTxt").style.top = $(".content").offsetHeight - $(".table").offsetTop + $(".content").offsetTop + "px";
				$(".asShownDisclaimerTxt").style.bottom = "0px";
				this.centerValignText(".asShownDisclaimerTxt span");
			break;
			case 'mathstack':
				//document.styleSheets[0].addRule('.mathstack .amountCondUpto:before','color: '+this.apr_up_to_copy+';');
				var style = (function() {
					// Create the <style> tag
					var style = document.createElement("style");
				    
					// WebKit hack
					style.appendChild(document.createTextNode(""));
				    
					// Add the <style> element to the page
					document.head.appendChild(style);
				      
					console.log(style.sheet.cssRules); // length is 0, and no rules
				    
					return style;
				})();
				    style.sheet.insertRule('.foo{color:red;}', 0);
				if(document.styleSheets[0].addRule){
					document.styleSheets[0].addRule('.mathstack .amountCondUpto:before','color: '+this.apr_up_to_copy+';');
				}else{
					style.sheet.insertRule('.mathstack .amountCondUpto:before {color: '+this.apr_up_to_copy+'}',0);
				}
				$(".content").style.marginTop = (this.asShownDisclaimerTxt.length > 0) ? "-6px" : "0";
				if(this.priceCount() > 1 && this.priceCount() < 4){
					for (var i = 1; i <= this.priceCount(); i++) {
						var ch = ( i * 3 ) - 1;
						console.log($(".copy:nth-child(" + ch + ")").offsetWidth + "/" + $(".copy:nth-child(" + ch + ") span").offsetWidth);
						$(".copy:nth-child(" + ch + ")").style.width = ($(".copy:nth-child(" + ch + ")").offsetWidth > $(".copy:nth-child(" + ch + ") span").offsetWidth) ? $(".copy:nth-child(" + ch + ") span").offsetWidth + 1 + "px" : $(".copy:nth-child(" + ch + ")").offsetWidth;
					}
					this.alignCenter(".equationWrapper",".content");
					this.alignCenter(".totalWrapper",".content");
					this.centerValignText(".table");
					// $(".totalWrapper").style.paddingLeft = (this.netCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0) ? "0px" : "25px";
					$(".equationWrapper").style.borderBottom = ($(".totalWrapper").offsetWidth > $(".equationWrapper").offsetWidth) ? "none" : '1px solid'+this.line_divider_color+'';
					$(".totalWrapper").style.borderTop = ($(".equationWrapper").offsetWidth > $(".totalWrapper").offsetWidth) ? "none" : '1px solid'+this.line_divider_color+'';
					
					$(".asShownDisclaimerTxt").style.left = (this.netCopy.toUpperCase().indexOf("DE AHORRO TOTAL") >= 0) ? $(".content").offsetLeft + $(".table").offsetLeft - 29 + "px" : $(".content").offsetLeft + $(".table").offsetLeft + "px";
					$(".asShownDisclaimerTxt").style.left = $(".content").offsetLeft + $(".table").offsetLeft + "px";
					$(".asShownDisclaimerTxt").style.top = $(".content").offsetHeight - $(".table").offsetTop + $(".content").offsetTop + "px";
					$(".asShownDisclaimerTxt").style.bottom = "0px";
					$(".asShownDisclaimerTxt").style.width = $(".content").offsetWidth + "px";
					$(".asShownDisclaimerTxt span").style.width = $(".table").offsetWidth + "px";

					this.centerValignText(".asShownDisclaimerTxt span");
					this.alignCenter(".asShownDisclaimerTxt span",".asShownDisclaimerTxt");
				} else {
					$(".copyCond").style.width = ($(".copyCond").offsetWidth > $(".copyCond span").offsetWidth) ? $(".copyCond span").offsetWidth + "px" : $(".copyCond").offsetWidth;
					this.alignCenter(".table",".content");
					this.centerValignText(".table");
					$(".content").style.marginLeft = (this.netCopy.toUpperCase().indexOf("TOTAL SAVINGS") >= 0) ? "15px" : "0";

					$(".asShownDisclaimerTxt").style.left = (this.netCopy.toUpperCase().indexOf("TOTAL SAVINGS") >= 0) ? $(".content").offsetLeft + $(".table").offsetLeft - 29 + "px" : $(".content").offsetLeft + $(".table").offsetLeft + "px";
					$(".asShownDisclaimerTxt").style.top = $(".content").offsetHeight - $(".table").offsetTop + $(".content").offsetTop + "px";
					$(".asShownDisclaimerTxt").style.bottom = "0px";
					$(".asShownDisclaimerTxt").style.width = $(".table").offsetWidth + (($(".content").offsetWidth - $(".table").offsetWidth) / 2) - 5 + "px";
					$(".asShownDisclaimerTxt span").style.width = $(".asShownDisclaimerTxt").offsetWidth + "px";
					this.centerValignText(".asShownDisclaimerTxt span");
				}
			break;
			case 'negmathstack':
				//document.styleSheets[0].addRule('.negmathstack .amountCondUpto:before','color: '+this.apr_up_to_copy+';');
				var style = (function() {
					// Create the <style> tag
					var style = document.createElement("style");
				    
					// WebKit hack
					style.appendChild(document.createTextNode(""));
				    
					// Add the <style> element to the page
					document.head.appendChild(style);
				      
					console.log(style.sheet.cssRules); // length is 0, and no rules
				    
					return style;
				})();
				    style.sheet.insertRule('.foo{color:red;}', 0);
				if(document.styleSheets[0].addRule){
					document.styleSheets[0].addRule('.negmathstack .amountCondUpto:before','color: '+this.apr_up_to_copy+';');
				}else{
					style.sheet.insertRule('.negmathstack .amountCondUpto:before {color: '+this.apr_up_to_copy+'}',0);
				}
				
				if(this.priceCount() > 1 && this.priceCount() < 4){
					for (var i = 1; i <= this.priceCount(); i++) {
						var ch = ( i * 3 ) - 1;
						console.log($(".copy:nth-child(" + ch + ")").offsetWidth + "/" + $(".copy:nth-child(" + ch + ") span").offsetWidth);
						$(".copy:nth-child(" + ch + ")").style.width = ($(".copy:nth-child(" + ch + ")").offsetWidth > $(".copy:nth-child(" + ch + ") span").offsetWidth) ? $(".copy:nth-child(" + ch + ") span").offsetWidth + "px" : $(".copy:nth-child(" + ch + ")").offsetWidth;
					}
					$(".content").style.marginTop = (this.asShownDisclaimerTxt.length > 0) ? "-6px" : "0";
					this.alignCenter(".equationWrapper",".content");
					this.alignCenter(".totalWrapper",".content");
					this.centerValignText(".table");
					// $(".totalWrapper").style.paddingLeft = (this.netCopy.toUpperCase().indexOf("TOTAL SAVING") >= 0 || $('.upTo').innerHTML.toUpperCase().indexOf("AS LOW AS") >= 0) ? "0px" : "25px";
					$(".equationWrapper").style.borderBottom = ($(".totalWrapper").offsetWidth > $(".equationWrapper").offsetWidth) ? "none" : '1px solid'+this.line_divider_color+'';
					$(".totalWrapper").style.borderTop = ($(".equationWrapper").offsetWidth > $(".totalWrapper").offsetWidth) ? "none" : '1px solid'+this.line_divider_color+'';

					$(".asShownDisclaimerTxt").style.left = $(".content").offsetLeft + $(".table").offsetLeft + "px";
					$(".asShownDisclaimerTxt").style.top = $(".content").offsetHeight - $(".table").offsetTop + $(".content").offsetTop + "px";
					$(".asShownDisclaimerTxt").style.bottom = "0px";
					$(".asShownDisclaimerTxt").style.width = $(".content").offsetWidth + "px";
					$(".asShownDisclaimerTxt span").style.width = $(".table").offsetWidth + "px";

					this.centerValignText(".asShownDisclaimerTxt span");
					this.alignCenter(".asShownDisclaimerTxt span",".asShownDisclaimerTxt");
				} else {
					$(".copyCond").style.width = ($(".copyCond").offsetWidth > $(".copyCond span").offsetWidth) ? $(".copyCond span").offsetWidth + "px" : $(".copyCond").offsetWidth;
					this.alignCenter(".table",".content");
					this.centerValignText(".table");
					$(".content").style.marginTop = (this.asShownDisclaimerTxt.length > 0) ? "-3px" : "2px";
					$(".content").style.marginLeft = (this.netCopy.toUpperCase().indexOf("TOTAL SAVINGS") >= 0) ? "15px" : "0";

					$(".asShownDisclaimerTxt").style.left = (this.netCopy.toUpperCase().indexOf("TOTAL SAVINGS") >= 0) ? $(".content").offsetLeft + $(".table").offsetLeft - 29 + "px" : $(".content").offsetLeft + $(".table").offsetLeft + "px";
					$(".asShownDisclaimerTxt").style.top = $(".content").offsetHeight - $(".table").offsetTop + $(".content").offsetTop + "px";
					$(".asShownDisclaimerTxt").style.bottom = "0px";
					$(".asShownDisclaimerTxt").style.width = $(".table").offsetWidth + (($(".content").offsetWidth - $(".table").offsetWidth) / 2) - 5 + "px";
					$(".asShownDisclaimerTxt span").style.width = $(".asShownDisclaimerTxt").offsetWidth + "px";
					this.centerValignText(".asShownDisclaimerTxt span");
				}
			break;
			case 'leasesignndrive':
				$(".table").style.top = "0";
				this.alignCenter(".leasesignndrive .table",".container");
				$(".leasesignndrive .table").style.marginLeft = $(".leasesignndrive .table").offsetLeft + 92 + "px";
				$(".leasesignndrive .table").style.marginTop = 5 + "px";
			break;
		    case 'leasesignndrivecashback':
				$(".table").style.top = "0";
				this.alignCenter(".leasesignndrivecashback .table",".container");
				$(".leasesignndrivecashback .table").style.marginLeft = $(".leasesignndrivecashback .table").offsetLeft + 92 + "px";
				$(".leasesignndrivecashback .table").style.marginTop = 5 + "px";
			break;
		}
		if(this.salesEventImg.length > 0 && this.salesEventImg.indexOf("Filler") < 0 && this.salesEventImg.indexOf("filler") < 0){
            if(this.salesEvent == "l4l"){
				if(this.offer == "lease" || this.offer == "leasecash") {
					$(".logo").innerHTML += "<div class='salesLogo'></div>";
					$(".salesLogo").style.backgroundImage = "url(" + dynamicContent.salesEvent[0].saleEventH5_728x90.Url + ")";
				}
			} else if (this.salesEvent == "minievent") {
				$(".logo").innerHTML += "<div class='salesLogo " + this.salesEvent + "'></div>";
				$(".salesLogo").style.backgroundImage = "url(" + dynamicContent.salesEvent[0].saleEventH5_728x90.Url + ")";
			} else if (this.salesEvent == "nosale") {
				//do nothing
			} else {
				$(".logo").style.opacity = 0;
				$(".cta.cta1").classList.add("salesEvent");
				$(".cta.cta2").classList.add("salesEvent");
			}
        }
	    this.centerValignText(".cta.cta1 .ctaTxt");
		this.centerValignText(".cta.cta2 .ctaTxt");
		//this.setVideo();
	};

	Offer.prototype.checkCurrency = function(amount) {
		amount = (amount.indexOf("$") >= 0) ? amount : "$" + amount;
		return this.superScriptText(amount,'$');
	};

	Offer.prototype.centerValignText = function(target) {
		console.log(arguments);
		$(target).style.height = $(target).offsetHeight + "px";
	    $(target).style.top = 0;
	    $(target).style.bottom = 0;
	    $(target).style.marginTop = "auto";
	    $(target).style.marginBottom = "auto";
	};

	Offer.prototype.centerHAlignText = function(target) {
		console.log(arguments);
		//$(target).style.width = $(target).offsetWidth + 3 + "px";
	    $(target).style.left = 0;
	    $(target).style.right = 0;
	    $(target).style.paddingLeft = 60 + "px";
	    $(target).style.marginLeft = "auto";
	    $(target).style.marginRight = "auto";
	};

	Offer.prototype.alignCenter = function(target, centerTo) {
		console.log(arguments);
		$(target).style.marginLeft = ($(centerTo).offsetWidth - $(target).offsetWidth) / 2 + "px";
	};

	Offer.prototype.superScriptBracket = function(text) {
		console.log(arguments);
		text = text.split("[").join("<sup>[");
		text = text.split("]").join("]</sup>");
		return text;
	};

	Offer.prototype.superScriptText = function(text,sup){
		console.log(arguments);
		var word = text;
		return word.split(sup).join("<sup>" + sup + "</sup>");
	};

	Offer.prototype.addClick = function() {
		$(".container").addEventListener('click',function(e){
			console.log(arguments + " : " + e.target.getAttribute("class"));
			switch(e.target.getAttribute("class")){
				case 'cta cta1 salesEvent':
                	Enabler.exitOverride("brochure_exit", dynamicContent.offers[0].baseBrochureUrl.Url + customUrl);
				break;
				case 'cta cta2 salesEvent':
                	Enabler.exitOverride("build_exit", dynamicContent.offers[0].baseBuildUrl.Url + customUrl2);
				break;
				case 'cta cta1':
                	Enabler.exitOverride("brochure_exit", dynamicContent.offers[0].baseBrochureUrl.Url + customUrl);
				break;
				case 'cta cta2':
                	Enabler.exitOverride("build_exit", dynamicContent.offers[0].baseBuildUrl.Url + customUrl2);
				break;
				case 'disclaimerTxt':
					$(".disclaimerBox").style.display = "block";
				break;
				case 'closeBtn':
					$(".disclaimerBox").style.display = "none";
				break;
		 		default:
					$(".disclaimerBox").style.display = "none";
					Enabler.exitOverride("Main_Exit", dynamicContent.offers[0].baseLandingUrl.Url + customUrl);
				break;
			}
		},false);
	};
})();



